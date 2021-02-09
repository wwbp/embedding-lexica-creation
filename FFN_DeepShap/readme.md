## FFN_DeepShap Lexicon Generation and Evaluation
- Run the attached ipynb to run the code and generate a lexicon (in the format specified in the readme for the project) from a given dataset
- The code relies on utils.py for its implementation of Feed Forward NN and training and evaluation functions
- Change the variable lexiconDataset in the second cell to change the train dataset from which lexicon will be created. Available datasets: nrc_joy, nrc_sadness, nec_surprise, nrc_fear, nrc_anger,empathy, yelp_subset, amazon_finefood_subset, amazon_toys_subset
- The cell which calculates the SHAP values would take considerable time to execute depending upon the training dataset, in the order of several days when using a GPU
-  Change the destination in the last cell before the 'Evaluation Cell' to set where the lexicon generated shoud be saved.
- The list of evaluation datasets can be updates in the variable dataList to evaluate a given lexicon against other datasets (review the readme of the project to see a list of evalutations which we ran)

