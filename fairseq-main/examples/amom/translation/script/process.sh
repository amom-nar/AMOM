TEXT=examples/amom/iwslt14.tokenized.de-en
data_dir=examples/amom/data-bin/
mkdir -p ${data_dir}

fairseq-preprocess --source-lang de --target-lang en \
    --joined-dictionary \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir  ${data_dir}/iwslt_de_en_raw\
    --workers 20