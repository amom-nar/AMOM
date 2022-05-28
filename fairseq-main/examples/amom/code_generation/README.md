# Code Generation

This example contains instructions for training a new non-autoregressive model for Py150 and GitHub-Java dataset with AMOM.

# Prepare Data
We can download the Py150 dataset [here](https://www.sri.inf.ethz.ch/py150) and GitHub-Java dataset [here](http://groups.inf.ed.ac.uk/cup/javaGithub).
Then we split the dataset following the description in our paper:
```
python script/split_data.py
```

# Binarize the data
We proprecess the data without applying BPE:
```
bash script/proprecess.sh
```

# Training
```
export CUDA_VISIBLE_DEVICES=0,1
fairseq-train ${data_dir} \
    --save-dir ${save_path} --ddp-backend=legacy_ddp \
    --task translation_lev --criterion nat_loss --arch amom_transformer \
    --optimizer adam --adam-betas '(0.9,0.999)' --lr 5e-5 --lr-scheduler inverse_sqrt --stop-min-lr 1e-09 
    --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --label-smoothing 0.1 --dropout 0.1 --weight-decay 0.01 --decoder-learned-pos --encoder-learned-pos \
    --apply-bert-init --fp16 --log-format simple --log-interval 100 --fixed-validation-seed 7 --batch-size-valid 50 \
    --max-tokens 8192 --save-interval-updates 10000 --max-update 300000 \
    --eval-bleu --eval-bleu-detok moses \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
```

# Inference
```
bash script/inference.sh
```