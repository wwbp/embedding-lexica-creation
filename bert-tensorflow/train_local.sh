export BERT_BASE_DIR=./uncased_L-4_H-512_A-8

python3.7 run_reg.py \
  --task_name=rate \
  --do_train=true \
  --do_eval=true \
  --data_dir=./data \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=5 \
  --output_dir=./output/ \
  --use_sigmoid_act=false
