input_dir = [where you save the splited data]
data_dir = [where to save the proprecessed data]
fairseq-preprocess --source-lang src --target-lang tgt --trainpref ${input_dir}/train \
    --validpref ${input_dir}/valid --testpref ${input_dir}/test --destdir ${data_dir} \
    --workers 64 --joined-dictionary