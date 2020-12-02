source ~/.bashrc
conda activate nlp

export rg="/export/c01/ztwu/empathy_dictionary/bert-pytorch"
export data="/export/c01/ztwu/empathy_dictionary/data"
export model="/export/c01/ztwu/empathy_dictionary/bert-pytorch/VAD/trainOnBert/do_train.True+early_stop.True+lr.1e-5+max_seq_length.128+num_train_epochs.50+num_warmup_steps.500+task.V+train_batch_size.32"

CUDA_VISIBLE_DEVICES=`free-gpu` PYTHONPATH=$rg python $rg/scripts/run_bert.py \
    --dataFolder=$data \
    --dataset=empathy \
    --output_dir=$model \
    --task=regression \
    --do_train \
    --model_kind=bert \
    --model=bert-base-uncased \
    --do_lower_case \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --num_train_epochs=50 \
    --lr=1e-5 \
    --num_warmup_steps=500 \
    --early_stop