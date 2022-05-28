# Neural Machine Translation

This example contains instructions for training a new NMT model with AMOM.

The following instructions can be used to train a [cmlm_transformer](https://aclanthology.org/D19-1633.pdf) with AMOM strategy.

# Prepare Data
Taken [IWSLT'14 German to English dataset](http://workshop2014.iwslt.org/downloads/proceeding.pdf) as an example, please following the steps list below.
* Download and preprocess the data: `bash script/prepare-iwslt14.sh`.
* Follow the instructions in [fairseq](https://github.com/pytorch/fairseq) to train an autoregressive model.
* Generate distilled target samples by the autoregressive model: `bash script/distill.sh`

# Training
* Train AMOM on the distill data.
```
fairseq-train ${data_dir} \
    --save-dir ${save_path} --ddp-backend=legacy_ddp \
    --task translation_lev --criterion nat_loss --arch amom_transformer_iwlst_de_en \
    --optimizer adam --adam-betas '(0.9,0.98)' --lr 0.0005 --apply-bert-init --fp16 \
    --lr-scheduler inverse_sqrt --stop-min-lr 1e-09 --warmup-updates 10000 --warmup-init-lr 1e-07 \
    --label-smoothing 0.1 --dropout 0.3 --weight-decay 0.01 --decoder-learned-pos --encoder-learned-pos \
    --log-format simple --log-interval 100 --fixed-validation-seed 7 --batch-size-valid 50 \
    --max-tokens 8192 --save-interval-updates 10000 --max-update 300000 \
    --eval-bleu --eval-bleu-remove-bpe --eval-bleu-detok moses \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \

```

# Inference
* Before inference, we use the average of best 5 checkpoints evaluted on valid BLEU score: `bash script/average.sh`.
* Generate the results.
```
fairseq-generate ${data_dir} \
   --path ${save_path}/checkpoint_average_bleu.pt \
   --task translation_lev --remove-bpe --fp16 \
   --source-lang de --target-lang en \
   --iter-decode-max-iter 9 --iter-decode-force-max-iter \
   --iter-decode-eos-penalty 0 --iter-decode-with-beam 3 --gen-subset test \
   --batch-size 50 > ${save_path}/result.gen \
```

