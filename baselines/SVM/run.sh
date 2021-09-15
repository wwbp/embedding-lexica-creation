export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/FFN_SVM

#Set up model kind, data folder and output folder here.
python baselines/SVM/main.py \
    --dataFolder=data \
    --output_dir=lexica/SVM