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

task evalOnBert : rg : data
  :: max_seq_length=@ :: train_batch_size=@ :: task=@ 
  :: dataset=@ :: model_kind=@ :: train_data=@
  :: pyenv=@ :: model=@
  :: .submitter=$grid :: .resource_flags=$gpuResourceFlags :: .action_flags=$gpuResourceFlags
{
  model="/home/zwu49/ztwu/embedding-lexica-creation/rerun/trainOnBert/dataset.*+max_seq_length.^+model_kind.#+pretrained.#-base+task.@+train_batch_size.!/best_model"
  model=${model//\*/$train_data}
  model=${model//^/$max_seq_length}
  model=${model//#/$model_kind}
  model=${model//@/$task}
  model=${model//!/$train_batch_size}
  
  CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/train_bert.py \
    --dataFolder=$data \
    --dataset=$dataset \
    --task=$task \
    --do_predict \
    --model_kind=$model_kind \
    --model=$model \
    --do_lower_case \
    --max_seq_length=$max_seq_length \
    --predict_batch_size=$train_batch_size
}

plan train {
  reach trainOnBert via (dataset: yelp_subset amazon_finefood_subset amazon_toys_subset nrc_joy nrc_anger nrc_fear nrc_sadness nrc_surprise) 
  * (task: classification) * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: roberta) * (pretrained: roberta-base)
  reach trainOnBert via (dataset: yelp_subset amazon_finefood_subset amazon_toys_subset nrc_joy nrc_anger nrc_fear nrc_sadness nrc_surprise) 
  * (task: classification) * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: distilbert) * (pretrained: distilbert-base-uncased)
}

plan eval_roberta {
  reach evalOnBert via (train_data: nrc_joy amazon_finefood_subset amazon_toys_subset yelp_subset) * (dataset: nrc_joy yelp_subset amazon_finefood_subset amazon_toys_subset song_joy emobank dialog_joy friends_joy) 
  * (task: classification) * (model_kind: roberta) * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_anger) * (dataset: nrc_anger song_anger dialog_anger friends_anger) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_sadness) * (dataset: nrc_sadness song_sadness dialog_sadness friends_sadness) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_fear) * (dataset: nrc_fear song_fear dialog_fear friends_fear) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_surprise) * (dataset: nrc_surprise song_surprise dialog_surprise friends_surprise) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 128) * (train_batch_size: 32)
}

plan eval_distil {
  reach evalOnBert via (train_data: nrc_joy amazon_finefood_subset amazon_toys_subset yelp_subset) * (dataset: nrc_joy yelp_subset amazon_finefood_subset amazon_toys_subset song_joy emobank dialog_joy friends_joy) 
  * (task: classification) * (model_kind: distilbert) * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_anger) * (dataset: nrc_anger song_anger dialog_anger friends_anger) * (task: classification) * (model_kind: distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_sadness) * (dataset: nrc_sadness song_sadness dialog_sadness friends_sadness) * (task: classification) * (model_kind: distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_fear) * (dataset: nrc_fear song_fear dialog_fear friends_fear) * (task: classification) * (model_kind: distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_surprise) * (dataset: nrc_surprise song_surprise dialog_surprise friends_surprise) * (task: classification) * (model_kind: distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
}