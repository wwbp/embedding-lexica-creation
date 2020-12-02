task trainOnBert : rg : data
  > model="best_model"
  :: max_seq_length=@ :: train_batch_size=@ :: task=@ 
  :: dataset=@ :: model_kind=@ :: pretrained=@
  :: pyenv=@
  :: .submitter=$grid :: .resource_flags=$gpuResourceFlags :: .action_flags=$gpuResourceFlags
{
  mkdir -p $model
  CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/train_bert.py \
    --dataFolder=$data \
    --dataset=$dataset \
    --output_dir=$model \
    --task=$task \
    --do_train \
    --model_kind=$model_kind \
    --model=$pretrained \
    --do_lower_case \
    --max_seq_length=$max_seq_length \
    --train_batch_size=$train_batch_size \
    --num_train_epochs=50 \
    --early_stop
}

plan PilotOnBert {
  reach trainOnBert via (dataset: empathy distress emobank_V emobank_A emobank_D yelp_subset) 
  * (task: classification regression) * (max_seq_length: 128) * (train_batch_size: 32) 
  * (model_kind: bert) * (pretrained: bert-base-uncased)
  reach trainOnBert via (dataset: empathy distress emobank_V emobank_A emobank_D yelp_subset) 
  * (task: classification regression) * (max_seq_length: 128) * (train_batch_size: 32) 
  * (model_kind: distilbert) * (pretrained: distilbert-base-uncased)
  reach trainOnBert via (dataset: yelp_subset) * (task: classification regression) * (max_seq_length: 256) 
  * (train_batch_size: 16) * (model_kind: bert) * (pretrained: bert-base-uncased)
  reach trainOnBert via (dataset: yelp_subset) * (task: classification regression) * (max_seq_length: 256) 
  * (train_batch_size: 16) * (model_kind: distilbert) * (pretrained: distilbert-base-uncased)
  reach trainOnBert via (dataset: yelp_subset) * (task: classification regression) * (max_seq_length: 512) 
  * (train_batch_size: 8) * (model_kind: bert) * (pretrained: bert-base-uncased)
  reach trainOnBert via (dataset: yelp_subset) * (task: classification regression) * (max_seq_length: 512) 
  * (train_batch_size: 8) * (model_kind: distilbert) * (pretrained: distilbert-base-uncased)
  reach trainOnBert via (dataset: nrc_joy nrc_sadness nrc_fear nrc_anger nrc_surprise) * (task: classification) 
  * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: bert) * (pretrained: bert-base-uncased)
  reach trainOnBert via (dataset: nrc_joy nrc_sadness nrc_fear nrc_anger nrc_surprise) * (task: classification) 
  * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: distilbert) * (pretrained: distilbert-base-uncased)
}
