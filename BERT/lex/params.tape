global {
  # Model architecture
  task=(task: regression classification)
  model_kind=(model_kind: bert distilbert roberta)
  pretrained=(pretrained: bert-base-uncased distilbert-base-uncased roberta-base)

  # training hyper-parameters
  max_seq_length=(max_seq_length: 128 256 512 200)
  train_batch_size=(train_batch_size: 8 16 32 64 128)
  eval_batch_size=(eval_batch_size: 8 16 32)
  predict_batch_size=(predict_batch_size: 8 16 32)
  lr=(lr: 3e-5 1e-5 5e-5)
  num_train_epochs=(num_train_epochs: 10 50 100)
  num_warmup_steps=(num_warmup_steps: 0 100 500 1000 2000 10000 5000)
  patience=(patience: 3 5 7)
  delta=(delta: 0 0.1 0.2)

  # dataset
  dataset=(dataset: empathy distress emobank_V emobank_A emobank_D yelp_subset nrc_joy nrc_sadness nrc_fear nrc_anger nrc_surprise amazon_finefood_subset amazon_toys_subset)
}
