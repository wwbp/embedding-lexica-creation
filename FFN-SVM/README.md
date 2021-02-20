##### Environment Setup

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/FFN-SVM
```

##### Download Tokenizer
```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip -o crawl-300d-2M.vec.zip
python -m spacy init-model -v crawl-300d-2M.vec en fasttext
```

##### Lexicon Generation

```bash
python FFN-SVM/[model_name]/generateLexicon.py --dataFolder [dataFolder] --output_dir [output_dir]
```

