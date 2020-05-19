export BERT_BASE_DIR=Desktop/JHU/Word/uncased_L-4_H-512_A-8

python3.7 Desktop/JHU/Word/empathy_dictionary/bert-tensorflow/run_reg.py \
  --task_name=rate \
  --do_predict=true \
  --data_dir=Desktop/JHU/Word/data/ \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=Desktop/JHU/Word/out/trainOnBert/batch_size.64+do_eval.true+do_train.true+learning_rate.1e-5+max_seq_length.256+num_epochs.800+task.rate+use_sigmoid_act.false/best_model \
  --max_seq_length=128 \
  --output_dir=Desktop/JHU/Word/ \
  --use_sigmoid_act=false
