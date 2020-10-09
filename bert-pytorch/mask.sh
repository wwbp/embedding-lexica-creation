source ~/.bashrc
conda activate nlp

export rg="/export/c01/ztwu/empathy_dictionary/bert-pytorch"
export data="/export/c01/ztwu/empathy_dictionary/data"
export model="/export/c01/ztwu/empathy_dictionary/bert-pytorch/stars/trainOnBert/early_stop.True+k_fold.0+lr.1e-5+max_seq_length.256+model_kind.distilbert+num_train_epochs.50+num_warmup_steps.0+pretrained.distilbert-base-uncased+task.stars+train_batch_size.16"

CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/mask.py \
    --data=$data/df1M.tsv \
    --task=stars \
    --model_kind=bert \
    --if_bert_embedding=True \
    --bert_model=$model/best_model \
    --tokenizer=$model/best_model \
    --early_stop=True \
    --output=$rg/lexicon_stars_256.csv \
    --patience=16 \
    --lr=1e-5 \
    --dropout=0.5 \
    --gold_word=$data/2000.tsv