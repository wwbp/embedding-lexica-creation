global {
  # Model architecture
  task=(task: empathy distress V A D)
  do_train=(do_train: True False)
  do_predict=(do_predict: True False)

  # training hyper-parameters
  max_seq_length=(max_seq_length: 128)
  train_batch_size=(train_batch_size: 32 64 128)
  eval_batch_size=(eval_batch_size: 8 16 32)
  predict_batch_size=(predict_batch_size: 8 16 32)
  lr=(lr: 3e-5 1e-5 5e-5)
  num_train_epochs=(num_train_epochs: 10 50 100)
  num_warmup_steps=(num_warmup_steps: 0 100 500 1000 2000 10000 5000)
  early_stop=(early_stop: True False)
  patience=(patience: 3 5 7)
  delta=(delta: 0 0.1 0.2)
}
