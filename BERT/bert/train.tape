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
  :: pyenv=@
  :: .submitter=$grid :: .resource_flags=$gpuResourceFlags :: .action_flags=$gpuResourceFlags
{
  model="/home/zwu49/ztwu/empathy_dictionary/roberta/trainOnBert/dataset.*+max_seq_length.^+model_kind.#+pretrained.#-base-uncased+task.@+train_batch_size.!"
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
    --model=$model/best_model \
    --do_lower_case \
    --max_seq_length=$max_seq_length \
    --predict_batch_size=$train_batch_size \
    --test_set
}

plan PilotOnBert {
  reach trainOnBert via (dataset: empathy distress emobank_V emobank_A emobank_D) 
  * (task: classification regression) * (max_seq_length: 128) * (train_batch_size: 32) 
  * (model_kind: bert) * (pretrained: bert-base-uncased)
  reach trainOnBert via (dataset: empathy distress emobank_V emobank_A emobank_D) 
  * (task: classification regression) * (max_seq_length: 128) * (train_batch_size: 32) 
  * (model_kind: distilbert) * (pretrained: distilbert-base-uncased)
  reach trainOnBert via (dataset: yelp_subset) * (task: classification regression) * (max_seq_length: 256) 
  * (train_batch_size: 16) * (model_kind: bert) * (pretrained: bert-base-uncased)
  reach trainOnBert via (dataset: yelp_subset) * (task: classification regression) * (max_seq_length: 256) 
  * (train_batch_size: 16) * (model_kind: distilbert) * (pretrained: distilbert-base-uncased)
  #reach trainOnBert via (dataset: yelp_subset) * (task: classification regression) * (max_seq_length: 512) 
  #* (train_batch_size: 8) * (model_kind: bert) * (pretrained: bert-base-uncased)
  #reach trainOnBert via (dataset: yelp_subset) * (task: classification regression) * (max_seq_length: 512) 
  #* (train_batch_size: 8) * (model_kind: distilbert) * (pretrained: distilbert-base-uncased)
  reach trainOnBert via (dataset: nrc_joy nrc_sadness nrc_fear nrc_anger nrc_surprise) * (task: classification) 
  * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: bert) * (pretrained: bert-base-uncased)
  reach trainOnBert via (dataset: nrc_joy nrc_sadness nrc_fear nrc_anger nrc_surprise) * (task: classification) 
  * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: distilbert) * (pretrained: distilbert-base-uncased)
  reach trainOnBert via (dataset: amazon_finefood_subset amazon_toys_subset) * (task: classification regression) * (max_seq_length: 128) 
  * (train_batch_size: 32) * (model_kind: bert) * (pretrained: bert-base-uncased)
  reach trainOnBert via (dataset: amazon_finefood_subset amazon_toys_subset) * (task: classification regression) * (max_seq_length: 128) 
  * (train_batch_size: 32) * (model_kind: distilbert) * (pretrained: distilbert-base-uncased)
  reach trainOnBert via (dataset: yelp_subset) * (task: classification regression) * (max_seq_length: 200) 
  * (train_batch_size: 16) * (model_kind: bert) * (pretrained: bert-base-uncased)
  reach trainOnBert via (dataset: yelp_subset) * (task: classification regression) * (max_seq_length: 200) 
  * (train_batch_size: 16) * (model_kind: distilbert) * (pretrained: distilbert-base-uncased)
}

plan eval_test {
  reach evalOnBert via (train_data: nrc_joy) * (dataset: nrc_joy empathy yelp_subset amazon_finefood_subset amazon_toys_subset) * (task: classification) * (model_kind: bert distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_anger) * (dataset: nrc_anger) * (task: classification) * (model_kind: bert distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_sadness) * (dataset: nrc_sadness) * (task: classification) * (model_kind: bert distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_fear) * (dataset: nrc_fear) * (task: classification) * (model_kind: bert distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_surprise) * (dataset: nrc_surprise) * (task: classification) * (model_kind: bert distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: amazon_finefood_subset) * (dataset: nrc_joy empathy yelp_subset amazon_finefood_subset amazon_toys_subset) * (task: classification) * (model_kind: bert distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: yelp_subset) * (dataset: nrc_joy empathy yelp_subset amazon_finefood_subset amazon_toys_subset) * (task: classification) * (model_kind: bert distilbert)
  * (max_seq_length: 256) * (train_batch_size: 16)
  reach evalOnBert via (train_data: amazon_toys_subset) * (dataset: nrc_joy empathy yelp_subset amazon_finefood_subset amazon_toys_subset) * (task: classification) * (model_kind: bert distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: empathy) * (dataset: nrc_joy empathy yelp_subset amazon_finefood_subset amazon_toys_subset) * (task: classification) * (model_kind: bert distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: yelp_subset) * (dataset: nrc_joy empathy yelp_subset amazon_finefood_subset amazon_toys_subset) * (task: classification) * (model_kind: bert distilbert)
  * (max_seq_length: 200) * (train_batch_size: 16)
}

plan temp {
  reach evalOnBert via (train_data: yelp_subset) * (dataset: nrc_joy empathy yelp_subset amazon_finefood_subset amazon_toys_subset) * (task: classification) * (model_kind: bert distilbert)
  * (max_seq_length: 200) * (train_batch_size: 16)
}

plan eval_other {
  reach evalOnBert via (train_data: nrc_joy empathy amazon_finefood_subset amazon_toys_subset) * (dataset: song_joy emobank_V dialog_joy friends_joy) 
  * (task: classification) * (model_kind: bert distilbert) * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_anger) * (dataset: song_anger dialog_anger friends_anger) * (task: classification) * (model_kind: bert distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_fear) * (dataset: song_fear dialog_fear friends_fear) * (task: classification) * (model_kind: bert distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_sadness) * (dataset: song_sadness dialog_sadness friends_sadness) * (task: classification) * (model_kind: bert distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_surprise) * (dataset: song_surprise dialog_surprise friends_surprise) * (task: classification) * (model_kind: bert distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: yelp_subset) * (dataset: song_joy emobank_V dialog_joy friends_joy) 
  * (task: classification) * (model_kind: bert distilbert) * (max_seq_length: 256) * (train_batch_size: 16)
  reach evalOnBert via (train_data: yelp_subset) * (dataset: song_joy emobank_V dialog_joy friends_joy) 
  * (task: classification) * (model_kind: bert distilbert) * (max_seq_length: 200) * (train_batch_size: 16)
}

plan RobertaTrain {
  reach trainOnBert via (dataset: yelp_subset) * (task: classification) * (max_seq_length: 200) 
  * (train_batch_size: 16) * (model_kind: roberta) * (pretrained: roberta-base)
  reach trainOnBert via (dataset: nrc_joy nrc_sadness nrc_fear nrc_anger nrc_surprise) * (task: classification) 
  * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: roberta) * (pretrained: roberta-base)
  reach trainOnBert via (dataset: amazon_finefood_subset amazon_toys_subset empathy) * (task: classification) 
  * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: roberta) * (pretrained: roberta-base)
}

plan RobertaEvalTest {
  reach evalOnBert via (train_data: nrc_joy) * (dataset: nrc_joy empathy yelp_subset amazon_finefood_subset amazon_toys_subset) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_anger) * (dataset: nrc_anger) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_sadness) * (dataset: nrc_sadness) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_fear) * (dataset: nrc_fear) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_surprise) * (dataset: nrc_surprise) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: amazon_finefood_subset) * (dataset: nrc_joy empathy yelp_subset amazon_finefood_subset amazon_toys_subset) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: amazon_toys_subset) * (dataset: nrc_joy empathy yelp_subset amazon_finefood_subset amazon_toys_subset) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: empathy) * (dataset: nrc_joy empathy yelp_subset amazon_finefood_subset amazon_toys_subset) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: yelp_subset) * (dataset: nrc_joy empathy yelp_subset amazon_finefood_subset amazon_toys_subset) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 200) * (train_batch_size: 16)
}

plan RobertaEvalOther {
  reach evalOnBert via (train_data: nrc_joy empathy amazon_finefood_subset amazon_toys_subset) * (dataset: song_joy emobank_V dialog_joy friends_joy) 
  * (task: classification) * (model_kind: roberta) * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_anger) * (dataset: song_anger dialog_anger friends_anger) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_fear) * (dataset: song_fear dialog_fear friends_fear) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_sadness) * (dataset: song_sadness dialog_sadness friends_sadness) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: nrc_surprise) * (dataset: song_surprise dialog_surprise friends_surprise) * (task: classification) * (model_kind: roberta)
  * (max_seq_length: 128) * (train_batch_size: 32)
  reach evalOnBert via (train_data: yelp_subset) * (dataset: song_joy emobank_V dialog_joy friends_joy) 
  * (task: classification) * (model_kind: roberta) * (max_seq_length: 200) * (train_batch_size: 16)
}

plan Yelp_small {
  reach trainOnBert via (dataset: yelp_subset) * (task: classification) * (max_seq_length: 128) 
  * (train_batch_size: 32) * (model_kind: roberta) * (pretrained: roberta-base)
  reach trainOnBert via (dataset: yelp_subset) * (task: classification) * (max_seq_length: 128) 
  * (train_batch_size: 32) * (model_kind: distilbert) * (pretrained: distilbert-base-uncased)
}

plan yelp_eval {
  reach evalOnBert via (train_data: yelp_subset) * (dataset: nrc_joy empathy yelp_subset amazon_finefood_subset amazon_toys_subset) * (task: classification) * (model_kind: distilbert)
  * (max_seq_length: 128) * (train_batch_size: 32)
}

plan yelp_eval_other {
  reach evalOnBert via (train_data: yelp_subset) * (dataset: song_joy emobank_V dialog_joy friends_joy) 
  * (task: classification) * (model_kind: distilbert) * (max_seq_length: 128) * (train_batch_size: 32)
}

plan new_nrc {
  reach trainOnBert via (dataset: nrc_joy nrc_sadness nrc_fear nrc_anger nrc_surprise) * (task: classification) 
  * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: distilbert) * (pretrained: distilbert-base-uncased)
  reach trainOnBert via (dataset: nrc_joy nrc_sadness nrc_fear nrc_anger nrc_surprise) * (task: classification) 
  * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: roberta) * (pretrained: roberta-base)
}