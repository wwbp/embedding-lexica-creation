source ~/.bashrc
conda activate nlp

export rg="/home/zwu49/ztwu/empathy_dictionary/bert-pytorch"
export data="/home/zwu49/ztwu/empathy_dictionary/data"
export model="ztwu/empathy_dictionary/nrc_anger/trainOnBert/early_stop.True+k_fold.0+lr.1e-5+max_seq_length.128+model_kind.distilbert+num_train_epochs.50+num_warmup_steps.0+pretrained.distilbert-base-uncased+task.classification+train_batch_size.32"

CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/mask.py \
    --data=$data//NRCData/msgs_tec.csv \
    --task=classification-anger \
    --model_kind=distilbert \
    --if_bert_embedding=True \
    --bert_model=$model/best_model \
    --tokenizer=$model/best_model \
    --early_stop=True \
    --output=$rg/anger_distilbert_mask.csv \
    --patience=16 \
    --lr=1e-5 \
    --dropout=0.5