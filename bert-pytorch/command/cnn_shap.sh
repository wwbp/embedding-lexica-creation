source ~/.bashrc
conda activate nlp

export rg="/home/zwu49/ztwu/empathy_dictionary/bert-pytorch"
export data="/home/zwu49/ztwu/empathy_dictionary/data"
export model="ztwu/empathy_dictionary/nrc_anger/trainOnBert/early_stop.True+k_fold.0+lr.1e-5+max_seq_length.128+model_kind.bert+num_train_epochs.50+num_warmup_steps.0+pretrained.bert-base-uncased+task.classification+train_batch_size.32"

CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/run_shap_cnn.py \
    --data=$data/NRCData/msgs_tec.csv \
    --task=classification-anger \
    --model=FFN \
    --model_kind=bert \
    --if_bert_embedding=True \
    --bert_model=$model/best_model \
    --tokenizer=$model/best_model \
    --early_stop=True \
    --output=$rg/anger_bert.csv \
    --patience=7 \
    --lr=1e-5 \
    --dropout=0.2 \
    --gold_word=$data/NRCData/NRC_ht_emotion_sentiment.csv