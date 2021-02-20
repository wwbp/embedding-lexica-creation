export rg=$(pwd)/BERT
export data=[dataFolder] #Define data folder route
export model=[model_dir] #Define the model location
export output_dir=[output_dir] #Define the location where the lexicon should be saved

CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/partition_shap.py \
    --dataFolder=$data \
    --dataset=nrc_surprise \
    --output_dir=$output_dir \
    --task=classification \
    --model_kind=distilbert \
    --model=$model/best_model \
    --tokenizer=$model/best_model \
    --max_seq_length=128 \
    --do_lower_case \
    --do_alignment