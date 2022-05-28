
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""

from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import NATransformerModel
from fairseq.utils import new_arange
import torch
import random
from contextlib import contextmanager


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
            (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


@contextmanager
def torch_seed(seed):
    state = torch.random.get_rng_state()
    state_cuda = torch.cuda.random.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        torch.cuda.random.set_rng_state(state_cuda)

#adaptive_X_masking
def adaptive_inputmask(target_tokens, y_mask, pad, bos, eos, unk):  
    target_masks = (
            target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
    )
    target_score = target_tokens.clone().float().uniform_()
    target_score.masked_fill_(~target_masks, 2.0)
    target_length = target_masks.sum(1).float()
    target_length = target_length * (-y_mask * 0.2 + 0.3)  # the mapping function
    # target_length = target_length + 1
    _, target_rank = target_score.sort(1)
    target_cutoff = new_arange(target_rank) < target_length[:, None].long()
    prev_target_tokens = target_tokens.masked_fill(
        target_cutoff.scatter(1, target_rank, target_cutoff), unk
    )
    return prev_target_tokens


@register_model("amom_transformer")
class AMOMNATransformerModel(NATransformerModel):
    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."
        pad = self.tgt_dict.pad()
        bos = self.tgt_dict.bos()
        eos = self.tgt_dict.eos()
        unk = self.tgt_dict.unk()

        rand_seed = random.randint(0, 19260817)

        # adaptive X masking
        word_ins_mask = prev_output_tokens.eq(self.unk)  
        y_mask = word_ins_mask.sum(1).float() / word_ins_mask.size(1) 
        src_tokens_ada = adaptive_inputmask(src_tokens, y_mask=y_mask, pad=pad, bos=bos, eos=eos, unk=unk)  
        encoder_out_ada = self.encoder(src_tokens_ada, src_lengths=src_lengths, **kwargs)  

        # length prediction
        length_out = self.decoder.forward_length(
            normalize=False, encoder_out=encoder_out_ada
        )
        length_tgt = self.decoder.forward_length_prediction(
            length_out, encoder_out_ada, tgt_tokens
        )

        # decoding
        with torch_seed(rand_seed):
            word_ins_out = self.decoder(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out_ada,
            )

        
        # correction ratio
        pred_token = word_ins_out.argmax(-1)
        right_num = ((pred_token == tgt_tokens) & word_ins_mask).sum(1)  
        tot_num = word_ins_mask.sum(1)  
        b = 0.6 * (right_num / tot_num) + 0.2  
        b = b.unsqueeze(-1)  
        pred_tokens_ada = prev_output_tokens.masked_fill(word_ins_mask, 0) + pred_tokens.masked_fill(
            ~word_ins_mask, 0)

        target_masks = (
                pred_tokens_ada.ne(pad) & pred_tokens_ada.ne(bos) & pred_tokens_ada.ne(eos)
        )

        # adaptive Y masking
        target_score = pred_tokens.clone().float().uniform_()
        target_score.masked_fill_(~target_masks, 2.0)  
        target_mask_ymask = ((target_score <= (1 - b)) & (word_ins_mask))  
        target_mask_yobs = ((target_score <= b) & (~word_ins_mask))  
        target_mask = (target_mask_ymask | target_mask_yobs)  
        prev_output_tokens_aday = pred_tokens_ada.masked_fill(target_mask, unk)
        word_ins_mask_aday = target_mask  

        # adaptive X masking
        y_mask_aday = word_ins_mask_aday.sum(1).float() / word_ins_mask_aday.size(1)  
        src_tokens_aday = adaptive_inputmask(src_tokens, y_mask=y_mask_aday, pad=pad, bos=bos, eos=eos, unk=unk)  
        encoder_out_aday = self.encoder(src_tokens_aday, src_lengths=src_lengths, **kwargs)  

        with torch_seed(rand_seed):
            word_ins_out_aday = self.decoder(
                normalize=False,
                prev_output_tokens=prev_output_tokens_aday,
                encoder_out=encoder_out_aday,
            )

        return {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": word_ins_mask,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "word_ins_aday": {
                "out": word_ins_out_aday,
                "tgt": tgt_tokens,
                "mask": word_ins_mask_aday,
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            },
            "length": {
                "out": length_out,
                "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor,
            },
        }

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):

        step = decoder_out.step
        max_step = decoder_out.max_step

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.eq(self.unk)
        _scores, _tokens = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
        ).max(-1)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])

        if history is not None:
            history.append(output_tokens.clone())

        # skeptical decoding (depend on the maximum decoding steps.)
        if (step + 1) < max_step:
            skeptical_mask = _skeptical_unmasking(
                output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
            )

            output_tokens.masked_fill_(skeptical_mask, self.unk)
            output_scores.masked_fill_(skeptical_mask, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )


@register_model_architecture("amom_transformer", "amom_transformer")
def cmlm_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.ngram_predictor = getattr(args, "ngram_predictor", 1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture("amom_transformer", "amom_transformer_wmt_en_de")
def cmlm_wmt_en_de(args):
    cmlm_base_architecture(args)


@register_model_architecture("amom_transformer", "amom_transformer_iwlst_de_en")
def cmlm_iwlst_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    cmlm_base_architecture(args)

@register_model_architecture("amom_transformer", "amom_transformer_summarization")
def cmlm_iwlst_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4 * 768)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    cmlm_base_architecture(args)