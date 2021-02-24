##### Required python packages
```bash
pip install shap transformers spacy
```

##### Download Tokenizer
```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip -o crawl-300d-2M.vec.zip
python -m spacy init-model -v crawl-300d-2M.vec en fasttext
```

##### Train Model

```bash
bash BERT/command/train_bert.sh
```

##### Lexicon Generation
1. Mask
```bash
bash BERT/command/mask.sh
```

2. Partition Shap
```bash
bash BERT/command/partition_shap.sh
```

3. Deep Shap
```bash
bash BERT/command/deep_shap.sh
```

