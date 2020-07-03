source ~/.bashrc
conda activate nlp

export rg="/export/c01/ztwu/empathy_dictionary/bert-pytorch"
export data="/export/c01/ztwu/empathy_dictionary/data"
export model="/export/c01/ztwu/empathy_dictionary/bert-pytorch/out/trainOnBert/do_train.True+early_stop.True+lr.1e-5+max_seq_length.128+num_train_epochs.50+num_warmup_steps.1000+task.empathy+train_batch_size.32"

CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/run_shap.py \
    --data=$data/responses/data/messages.csv \
    --task=empathy \
    --model=$model/best_model \
    --tokenizer=$model/best_model \
    --early_stop=True \
    --output=$rg/lexicon.csv \
    --patience=20 \
    --lr=1e-5 \
    --dropout=0.5
    #--gold_word=$data/emobank/BRM-emot-submit.csv