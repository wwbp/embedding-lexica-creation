task trainOnBert : rg : data
  > model="best_model"
  :: max_seq_length=@ :: train_batch_size=@ :: lr=@
  :: num_train_epochs=@ :: num_warmup_steps=@
  :: task=@ :: early_stop=@ :: k_fold=@
  :: model_kind=@ :: pretrained=@
  :: pyenv=@
  :: .submitter=$grid :: .resource_flags=$gpuResourceFlags :: .action_flags=$gpuResourceFlags
{
  mkdir -p $model
  CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/regression.py \
    --task=$task \
    --do_train \
    --data_file=$data/NRCData/msgs_tec.csv \
    --model_kind=$model_kind \
    --model=$pretrained \
    --max_seq_length=$max_seq_length \
    --train_batch_size=$train_batch_size \
    --lr=$lr \
    --num_train_epochs=$num_train_epochs \
    --num_warmup_steps=$num_warmup_steps \
    --output_dir=$model \
    --early_stop \
    --k_fold=0 \
    --do_lower_case
}

plan PilotOnBert {
  reach trainOnBert via (task: classification) * (max_seq_length: 128) * (lr: 1e-5)
  * (num_train_epochs: 50) * (num_warmup_steps: 0) * (train_batch_size: 32) 
  * (model_kind: bert) * (pretrained: bert-base-uncased)
  #  reach trainOnBert via (task: stars) * (max_seq_length: 256) * (lr: 1e-5)
  #  * (num_train_epochs: 50) * (num_warmup_steps: 0) * (train_batch_size: 16) 
  #  * (model_kind: bert) * (pretrained: bert-base-uncased)
  #  reach trainOnBert via (task: stars) * (max_seq_length: 512) * (lr: 1e-5)
  #  * (num_train_epochs: 50) * (num_warmup_steps: 0) * (train_batch_size: 8) 
  #  * (model_kind: bert) * (pretrained: bert-base-uncased)
  reach trainOnBert via (task: classification) * (max_seq_length: 128) * (lr: 1e-5)
  * (num_train_epochs: 50) * (num_warmup_steps: 0) * (train_batch_size: 32) 
  * (model_kind: distilbert) * (pretrained: distilbert-base-uncased)
  #  reach trainOnBert via (task: stars) * (max_seq_length: 256) * (lr: 1e-5)
  #  * (num_train_epochs: 50) * (num_warmup_steps: 0) * (train_batch_size: 16) 
  #  * (model_kind: distilbert) * (pretrained: distilbert-base-uncased)
  #  reach trainOnBert via (task: stars) * (max_seq_length: 512) * (lr: 1e-5)
  #  * (num_train_epochs: 50) * (num_warmup_steps: 0) * (train_batch_size: 8) 
  #  * (model_kind: distilbert) * (pretrained: distilbert-base-uncased)
}
