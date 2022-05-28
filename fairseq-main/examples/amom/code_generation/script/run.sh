data_dir = [Path to your processed data]
save_path = [Path to save the model checkpoint]
export CUDA_VISIBLE_DEVICES=0,1
fairseq-train ${data_dir} \
    --save-dir ${save_path} --ddp-backend=legacy_ddp \
    --task translation_lev --criterion nat_loss --arch cmlm_transformer \
    --optimizer adam --adam-betas '(0.9,0.999)' --lr 5e-5 \
    --lr-scheduler inverse_sqrt --stop-min-lr 1e-09 --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --label-smoothing 0.1 --dropout 0.1 --weight-decay 0.01 --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --fp16 --log-format simple --log-interval 100 --fixed-validation-seed 7 --batch-size-valid 50 \
    --max-tokens 8192 --save-interval-updates 10000 --max-update 300000 \
    --eval-bleu -eval-bleu-detok moses \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \