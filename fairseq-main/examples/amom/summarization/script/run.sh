data_dir = [Path to your processed data]
save_path = [Path to saave the model checkpoint]
export CUDA_VISIBLE_DEVICES=0,1,2,3
fairseq-train ${data_dir} \
    --save-dir ${save_path} --ddp-backend=legacy_ddp \
    --task translation_lev --criterion nat_loss --arch amom_transformer_summarization \
    --optimizer adam --adam-betas '(0.9,0.999)' --adam-eps 1e-08 --lr 0.0001 --apply-bert-init --fp16 \
    --lr-scheduler inverse_sqrt --stop-min-lr 1e-09 --warmup-updates 10000 --warmup-init-lr 1e-07 --clip-norm 0.1 \
    --label-smoothing 0.1 --dropout 0.1 --weight-decay 0.01 --attention-dropout 0.1 --decoder-learned-pos --encoder-learned-pos \
    --log-format simple --log-interval 100 --fixed-validation-seed 7 --share-all-embeddings \
    --max-tokens 4096 --update-freq 2 --save-interval-updates 10000 --max-update 180000 \
    --required-batch-size-multiple 1 --skip-invalid-size-inputs-valid-test \