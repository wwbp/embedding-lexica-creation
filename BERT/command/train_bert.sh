export rg=$(pwd)/BERT
export data=[dataFolder] #Define data folder route
export model=[output_dir] #Define the location where the model should be saved

PYTHONPATH=$rg python $rg/scripts/train_bert.py \
    --dataFolder=$data \
    --dataset=yelp_subset \ #Define the dataset we use
    --output_dir=$model \
    --task=classification \
    --do_train \
    --model_kind=roberta \ #bert or distilbert
    --model=roberta-base \ #align with model_kind
    --do_lower_case \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --num_train_epochs=50 \
    --early_stop \
    --no_special_tokens #if you want to do a defalut training with special tokens, i.e. [cls], delete this line
