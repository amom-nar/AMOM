data_dir = [Path to your processed data]
save_path = [Path to your model checkpoint]
fairseq-generate ${data_dir} \
   --path ${save_path}/checkpoint_average_bleu.pt \
   --task translation_lev --remove-bpe --fp16 \
   --source-lang src --target-lang tgt \
   --iter-decode-max-iter 9 --iter-decode-force-max-iter \
   --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 --gen-subset test \
   --batch-size 50 > ${save_path}/result.gen \

bash fairseq-main/scripts/compound_split_bleu.sh ${save_path}/result.gen

# compute ES score
# python script/ES.py
