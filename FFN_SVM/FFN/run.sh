export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/FFN_SVM

python FFN_SVM/FFN/main.py \
    --dataFolder=data \
    --model_dir=FFN_model \
    --lexicon_dir=lexica/FFN \
    --train_model


