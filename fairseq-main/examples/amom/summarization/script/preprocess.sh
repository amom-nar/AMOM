wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

data_dir = [Path to your raw data]
cd ${data_dir}
for SPLIT in train val
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "${data_dir}/$SPLIT.$LANG" \
    --outputs "${data_dir}/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done


fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${data_dir}/train.bpe" \
  --validpref "${data_dir}/val.bpe" \
  --destdir "${data_dir}/xsum-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;