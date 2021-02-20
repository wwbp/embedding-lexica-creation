export rg="/export/c01/ztwu/empathy_dictionary/bert-pytorch"
export data="/export/c01/ztwu/empathy_dictionary/data"
export model="/export/c01/ztwu/empathy_dictionary/bert-pytorch/VAD/trainOnBert/do_train.True+early_stop.True+lr.1e-5+max_seq_length.128+num_train_epochs.50+num_warmup_steps.500+task.V+train_batch_size.32"

CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/regression.py \
    --task=V \
    --do_predict=true \
    --data_dir=$data/emobank \
    --model=$model/best_model \
    --tokenizer=$model/best_model \
    --max_seq_length=128