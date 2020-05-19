python3 regression.py \
  --task=empathy \
  --do_predict=true \
  --data_dir=./data \
  --model=./output_empathy \
  --tokenizer=./output_empathy \
  --max_seq_length=128 \
  --predict_batch_size=32