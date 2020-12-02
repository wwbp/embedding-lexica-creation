source ~/.bashrc
conda activate nlp

export rg="/home/zwu49/ztwu/empathy_dictionary/bert-pytorch"
export data="/home/zwu49/ztwu/empathy_dictionary/data"
export model="/home/zwu49/ztwu/empathy_dictionary/final/trainOnBert/dataset.nrc_surprise+max_seq_length.128+model_kind.distilbert+pretrained.distilbert-base-uncased+task.classification+train_batch_size.32"

CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/partition_shap.py \
    --dataFolder=$data \
    --dataset=nrc_surprise \
    --output_dir=/home/zwu49/ztwu/empathy_dictionary/final/lexicons \
    --task=classification \
    --model_kind=distilbert \
    --model=$model/best_model \
    --tokenizer=$model/best_model \
    --max_seq_length=128