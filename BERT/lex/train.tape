task mask : rg : data
  :: max_seq_length=@ :: train_batch_size=@ :: task=@ 
  :: dataset=@ :: model_kind=@
  :: pyenv=@
  :: .submitter=$grid :: .resource_flags=$gpuResourceFlags :: .action_flags=$gpuResourceFlags
{ 
  model="/home/zwu49/ztwu/embedding-lexica-creation/rerun/trainOnBert/dataset.*+max_seq_length.^+model_kind.#+pretrained.#-base+task.@+train_batch_size.!"
  model=${model//\*/$dataset}
  model=${model//^/$max_seq_length}
  model=${model//#/$model_kind}
  model=${model//@/$task}
  model=${model//!/$train_batch_size}

  out="/home/zwu49/ztwu/embedding-lexica-creation/lexicon_rerun"
  
  if [ ! -d $out ]; then
    mkdir $out
  fi
  
  CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/mask.py \
    --dataFolder=$data \
    --dataset=$dataset \
    --output_dir=$out\
    --task=$task \
    --model_kind=$model_kind \
    --model=$model/best_model \
    --max_seq_length=$max_seq_length \
    --batch_size=$train_batch_size \
    --do_alignment
}

task ps : rg : data
  :: max_seq_length=@ :: task=@ :: train_batch_size=@
  :: dataset=@ :: model_kind=@
  :: pyenv=@
  :: .submitter=$grid :: .resource_flags=$gpuResourceFlags :: .action_flags=$gpuResourceFlags
{
  model="/home/zwu49/ztwu/embedding-lexica-creation/rerun/trainOnBert/dataset.*+max_seq_length.^+model_kind.#+pretrained.#-base+task.@+train_batch_size.!"
  model=${model//\*/$dataset}
  model=${model//^/$max_seq_length}
  model=${model//#/$model_kind}
  model=${model//@/$task}
  model=${model//!/$train_batch_size}

  out="/home/zwu49/ztwu/embedding-lexica-creation/lexicon_rerun"
  
  if [ ! -d $out ]; then
    mkdir $out
  fi

  CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/partition_shap.py \
    --dataFolder=$data \
    --dataset=$dataset \
    --output_dir=$out \
    --task=$task \
    --model_kind=$model_kind \
    --model=$model/best_model \
    --max_seq_length=$max_seq_length \
    --do_alignment
}

task ig : rg : data
  :: max_seq_length=@ :: task=@ :: train_batch_size=@
  :: dataset=@ :: model_kind=@
  :: pyenv=@
  :: .submitter=$grid :: .resource_flags=$gpuResourceFlags :: .action_flags=$gpuResourceFlags
{
  model="/home/zwu49/ztwu/embedding-lexica-creation/rerun/trainOnBert/dataset.*+max_seq_length.^+model_kind.#+pretrained.#-base+task.@+train_batch_size.!"
  model=${model//\*/$dataset}
  model=${model//^/$max_seq_length}
  model=${model//#/$model_kind}
  model=${model//@/$task}
  model=${model//!/$train_batch_size}

  out="/home/zwu49/ztwu/embedding-lexica-creation/lexicon_rerun"
  
  if [ ! -d $out ]; then
    mkdir $out
  fi

  CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/integrated_gradient.py \
    --dataFolder=$data \
    --dataset=$dataset \
    --output_dir=$out \
    --task=$task \
    --model_kind=$model_kind \
    --model=$model/best_model \
    --max_seq_length=$max_seq_length \
    --do_alignment
}

task deep : rg : data
  :: max_seq_length=@ :: task=@ :: train_batch_size=@
  :: dataset=@ :: model_kind=@
  :: pyenv=@
  :: .submitter=$grid :: .resource_flags=$gpuResourceFlags :: .action_flags=$gpuResourceFlags
{
  model="/home/zwu49/ztwu/embedding-lexica-creation/rerun/trainOnBert/dataset.*+max_seq_length.^+model_kind.#+pretrained.#-base+task.@+train_batch_size.!"
  model=${model//\*/$dataset}
  model=${model//^/$max_seq_length}
  model=${model//#/$model_kind}
  model=${model//@/$task}
  model=${model//!/$train_batch_size}

  out="/home/zwu49/ztwu/embedding-lexica-creation/lexicon_rerun"
  
  if [ ! -d $out ]; then
    mkdir $out
  fi

  CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/deepshap.py \
    --dataFolder=$data \
    --dataset=$dataset \
    --output_dir=$out \
    --task=$task \
    --model_kind=$model_kind \
    --model=$model/best_model \
    --max_seq_length=$max_seq_length \
    --do_alignment \
    --background_size=50
}

plan lex_ig_d {
  reach ig via (dataset: yelp_subset nrc_joy nrc_sadness nrc_fear nrc_anger nrc_surprise amazon_finefood_subset amazon_toys_subset) 
  * (task: classification) * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: distilbert)
}

plan lex_ig_r {
  reach ig via (dataset: yelp_subset nrc_joy nrc_sadness nrc_fear nrc_anger nrc_surprise amazon_finefood_subset amazon_toys_subset) 
  * (task: classification) * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: roberta)
}

plan lex_d {
  #reach mask via (dataset: yelp_subset nrc_joy nrc_sadness nrc_fear nrc_anger nrc_surprise amazon_finefood_subset amazon_toys_subset) 
  #* (task: classification) * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: distilbert)
  reach ps via (dataset: yelp_subset nrc_joy nrc_sadness nrc_fear nrc_anger nrc_surprise amazon_finefood_subset amazon_toys_subset) 
  * (task: classification) * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: distilbert)
}

plan lex_r {
  #reach mask via (dataset: yelp_subset nrc_joy nrc_sadness nrc_fear nrc_anger nrc_surprise amazon_finefood_subset amazon_toys_subset) 
  #* (task: classification) * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: roberta)
  reach ps via (dataset: yelp_subset nrc_joy nrc_sadness nrc_fear nrc_anger nrc_surprise amazon_finefood_subset amazon_toys_subset) 
  * (task: classification) * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: roberta)
}

plan lex_deep {
  #reach deep via (dataset: yelp_subset) * (task: classification regression) * (max_seq_length: 128) 
  #* (train_batch_size: 32) * (model_kind: bert distilbert)
  #reach deep via (dataset: yelp_subset) * (task: classification regression) * (max_seq_length: 256) 
  #* (train_batch_size: 16) * (model_kind: bert distilbert)
  #reach deep via (dataset: yelp_subset) * (task: classification regression) * (max_seq_length: 512) 
  #* (train_batch_size: 8) * (model_kind: bert distilbert)
  reach deep via (dataset: nrc_joy nrc_sadness nrc_fear nrc_anger nrc_surprise) * (task: classification) 
  * (max_seq_length: 128) * (train_batch_size: 32) * (model_kind: bert distilbert)
  reach deep via (dataset: empathy amazon_finefood_subset amazon_toys_subset) * (task: classification) * (max_seq_length: 128) 
  * (train_batch_size: 32) * (model_kind: bert distilbert)
}

plan debug {
  reach ps via (dataset: nrc_surprise) * (task: classification) * (max_seq_length: 128) 
  * (train_batch_size: 32) * (model_kind: distilbert)
}