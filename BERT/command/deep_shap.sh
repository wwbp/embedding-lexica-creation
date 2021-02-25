export rg=$(pwd)/BERT
export data=[dataFolder] #Define data folder route
export model=[model_dir] #Define the model location
export output_dir=[output_dir] #Define the location where the lexicon should be saved

PYTHONPATH=$rg python $rg/scripts/deepshap.py \
    --dataFolder=$data \
    --dataset=nrc_surprise \ #Choose datasets
    --output_dir=$output_dir \
    --task=classification \
    --model_kind=distilbert \ #Choose model kinds
    --model=$model/best_model \
    --tokenizer=$model/best_model \
    --max_seq_length=128 \
    --background_size=200 \
    --do_lower_case \
    --do_alignment