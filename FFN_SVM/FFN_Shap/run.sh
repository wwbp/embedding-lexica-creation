source .bashrc
cd ztwu/empathy_dictionary

conda activate nlp

export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/FFN_SVM

CUDA_VISIBLE_DEVICES=`free-gpu` python FFN_SVM/FFN_Shap/partition.py \
    --dataFolder=data \
    --output_dir=lexica/FFN_Partition_ParitionMasker \
    --masker=Partition