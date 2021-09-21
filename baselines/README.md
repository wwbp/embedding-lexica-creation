##### Download Tokenizer
```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip -o crawl-300d-2M.vec.zip
python -m spacy init-model -v crawl-300d-2M.vec en fasttext
```

##### Lexicon Generation (Univariant)
```bash
bash FFN_SVM/univariant/run.sh
```
you should get all the lexicon .csv files and one results.csv file for evaluation.

##### Lexicon Generation (SVM)

```bash
bash FFN_SVM/SVM/run.sh
```
you should get all the lexicon .csv files and one results.csv file for evaluation.

##### Model train Lexicon Generation (FFN)

```bash
bash FFN_SVM/FFN/run.sh
```
you should get all the models, lexicon .csv files and one results.csv file for evaluation.