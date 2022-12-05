##### Environment Setup

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/diversityLSTM
```

##### Data Preprocessing

```bash
bash diversityLSTM/Transparency/preprocess/run_preprocessing.sh
```

You should expect 25 .csv files and 25 .p files

##### Lexicon Generation

```bash
bash diversityLSTM/Transparency/run_[dataset_name]
```

##### Evaluation

```bash
python diversityLSTM/Transparency/evaluation/evaluate_model_lexicon.py \
    --dataFolder=[data location]
```

