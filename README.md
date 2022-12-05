# Inducing Generalizable and Interpretable Lexica

## Download data

```bash
wget http://wwbp.org/downloads/public_data/dataset-lexicon_project.zip
```

## Download Tokenizer
```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip -o crawl-300d-2M.vec.zip
python -m spacy init-model -v crawl-300d-2M.vec en fasttext
```

## Methods

Each method will have its own directory containing all the code necessary to reproduce the results.

- Baselines
  
  - Univariate
  
  - SVM
  
  - FFN

- diversityLSTM

- BERT

## Lexica

One directory for each method that contains all the lexica generated using the method using the following datasets

- yelp_subset train set
- amazon_finefood_subset train set
- amazon_toys_subset train set
- nrc_joy train set
- nrc_fear train set
- nrc_anger train set
- nrc_sadness train set
- nrc_surprise train set

For each lexicon, the name of the file is in the format of '[dataset]_[method].csv' where [method] can be [model]_[feature importance measure]. For example, 'yelp_subset_ffn_deepshap.csv'. The csv file contains three columns with name 'word, score, word_count' respectively, so that they can be easily fed into the lexica evaluation pipeline. Do not include index when writing out the csv file.

## Evaluation

It contains the lexica evaluation pipeline that can be used commonly across all methods (Roshan) and the evaluation results including F1, Accuracy and other metrics for both lexica and models.

- Positive/Negative evaluations
  - Train on:
    - yelp_subset train set
    - amazon_finefood_subset train set
    - amazon_toys_subset train set
    - nrc_joy train set
  - Evalutate on:
    - yelp_subset test set
    - amazon_finefood_subset test set
    - amazon_toys_subset test set
    - nrc_joy test set
    - song_joy
    - dialog_joy
    - friends_joy
    - emobank
- Emotional Label evaluations
  - Train on:
    - nrc_[EMO] train set
  - Evaluate on:
    - nrc_[EMO]  test set
    - song_[EMO] 
    - dialog_[EMO] 
    - friends_[EMO] 
  - Where [EMO] = {joy, fear, anger, sadness, surprise}

## Analysis

Analysis includes dataset information, statistical evaluation results and visualization. Instructions and comments are in the notebooks.

## Reference

If you find this repo useful for your research, please cite

```bash

```
