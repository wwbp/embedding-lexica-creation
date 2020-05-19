python3 regression.py \
  --task=empathy \
  --do_train=true \
  --data_dir=./data \
  --bert_model=bert-base-uncased \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --lr=2e-5 \
  --num_train_epochs=5 \
  --output_dir=./output/
