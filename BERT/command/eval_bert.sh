export rg=$(pwd)/BERT
export data=[dataFolder] #Define data folder route
export model=[output_dir] #Define the location where the model should be saved

PYTHONPATH=$rg python $rg/scripts/run_bert.py \
    --dataFolder=$data \
    --dataset=yelp_subset \ #Define the dataset we use
    --task=classification \
    --do_prediction \
    --model_kind=roberta \ #bert or distilbert
    --model=roberta-base \ #align with model_kind
    --do_lower_case \
    --max_seq_length=128 \
    --predict_batch_size=32 \
    --use_lr_model\ # whether use logistic regression in do_predition
    --no_special_tokens #should be consistent with your training config