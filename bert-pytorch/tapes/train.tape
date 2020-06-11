task trainOnBert : rg : data
  > model="best_model"
  :: max_seq_length=@ :: train_batch_size=@ :: lr=@
  :: num_train_epochs=@ :: num_warmup_steps=@
  :: task=@ :: do_train=@ :: early_stop=@
  :: pyenv=@
  :: .submitter=$grid :: .resource_flags=$gpuResourceFlags :: .action_flags=$gpuResourceFlags
{
  mkdir -p $model
  CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/regression.py \
    --task=$task \
    --do_train=$do_train \
    --data_dir=$data/responses/data \
    --model=bert-base-uncased \
    --max_seq_length=$max_seq_length \
    --train_batch_size=$train_batch_size \
    --lr=$lr \
    --num_train_epochs=$num_train_epochs \
    --num_warmup_steps=$num_warmup_steps \
    --output_dir=$model \
    --early_stop=$early_stop
}

plan PilotOnBert {
  reach trainOnBert via (task: empathy distress) * (do_train: True) * (max_seq_length: 128) * (lr: 1e-5)
  * (num_train_epochs: 50) * (num_warmup_steps: 5000) * (early_stop: True)
}
