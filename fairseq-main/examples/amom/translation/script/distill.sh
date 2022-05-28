export CUDA_VISIBLE_DEVICES=0
src=de
tgt=en
data_dir = [Path to your training data]
save_path = [Path to model checkpoint]
output = [Path to save data]
data_name=${src}-${tgt}
mkdir -p ${output}

fairseq-generate ${data_dir} \
    --gen-subset train \
    --path ${save_path} \
    -s ${src} -t ${tgt} \
    --batch-size 512 --beam 4 --remove-bpe \
    --results-path ${output}/train.kd.gen

grep ^S ${output}/generate-train.txt | cut -f2- > ${output}/train.kd.${src}
grep ^H ${output}/generate-train.txt | cut -f3- > ${output}/train.kd.${tgt}

raw=${output}/raw_${src}_${tgt}
mkdir ${raw}
mv ${output}/train.kd.${src} ${output}/train.kd.${tgt} ${raw}

ORI_TEXT = [Path to your raw data]
DISTIL_TEXT=${output}/distill_${src}_${tgt}

apply bpe using original code
mkdir ${DISTIL_TEXT}
cp ${ORI_TEXT}/code ${ORI_TEXT}/valid.${src} ${ORI_TEXT}/valid.${tgt} ${ORI_TEXT}/test.${src} ${ORI_TEXT}/test.${tgt} ${raw} 
cd ${raw}
subword-nmt apply-bpe -c code < train.kd.${src} > train.${src} && rm train.kd.${src}
subword-nmt apply-bpe -c code < train.kd.${tgt}> train.${tgt} && rm train.kd.${tgt} 

fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref train --validpref valid --testpref test \
    --destdir  ${output}/databin/distill_${src}_${tgt} \
    --srcdict ${data_dir}/dict.${src}.txt \
    --tgtdict ${data_dir}/dict.${tgt}.txt 
