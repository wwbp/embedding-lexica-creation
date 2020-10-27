source ~/.bashrc
conda activate nlp

export rg="/export/c01/ztwu/empathy_dictionary/bert-pytorch"
export data="/export/c01/ztwu/empathy_dictionary/data"
export model="/export/c01/ztwu/empathy_dictionary/bert-pytorch/VAD/trainOnBert/do_train.True+early_stop.True+lr.1e-5+max_seq_length.128+num_train_epochs.50+num_warmup_steps.100+task.D+train_batch_size.32"

CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/run_shap.py \
    --data=$data/emobank/emobank.csv \
    --task=D \
    --model=$model/best_model \
    --tokenizer=$model/best_model \
    --output=$rg/lexicon_bert_D.csv \
    --gold_word=$data/emobank/BRM-emot-submit.csv