data_dir = [Path to your raw data]
save_path = [Path to model checkpoint]
cp ${data_dir}/dict.source.txt  ${save_path}
python script/summarize.py \
  --model-dir ${save_path} \
  --model-file checkpoint_best.pt \
  --src ${data_dir}/test.source \
  --out ${save_path}/test.hypo \
  --xsum-kwargs

python script/inference.py