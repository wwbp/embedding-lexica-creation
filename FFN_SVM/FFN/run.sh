export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/FFN_SVM
export PYTHONPATH=$PYTHONPATH:$(pwd)/FFN_SVM/FFN

TASK = $1
if [ $TASK = "train" ]; then 
    python FFN_SVM/SVM/main.py \
        --dataFolder=data \
        --output_dir=FFN_model \
        --task=train
elif [ $TASK = "gen"]; then
    METHOD = $2
    if [ $METHOD = "partition" ]; then
        MASK = $3
        python FFN_SVM/SVM/main.py \
            --dataFolder=data \
            --output_dir=lexica/FFN \
            --task=generate \
            --method=$METHOD \
            --model_dir=FFN_model \
            --masker=$MASK
    else
        python FFN_SVM/SVM/main.py \
            --dataFolder=data \
            --output_dir=lexica/FFN \
            --task=generate \
            --method=$METHOD \
            --model_dir=FFN_model
    fi
fi


