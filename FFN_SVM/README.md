##### Download Tokenizer
```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip -o crawl-300d-2M.vec.zip
python -m spacy init-model -v crawl-300d-2M.vec en fasttext
```

##### Lexicon Generation

```bash
bash FFN_SVM/run.sh [model_name]
```
model_name can be chosen from FFN and SVM, and you should get all the lexicon .csv files and one results.csv file for evaluation.
