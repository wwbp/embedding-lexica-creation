export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/FFN_SVM

#Set up model kind, data folder and output folder here.
python baselines/univariant/main.py \
    --dataFolder=data \
    --output_dir=lexica/uni