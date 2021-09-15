import sys
sys.path.append("/home/zwu49/ztwu/masker/shap_maskers")

import torch
import spacy

from utils import testModel_FFN, generateFastTextData_Spacy, Dataset


model = torch.load("/home/zwu49/ztwu/masker/ffn_models/yelp_subset.pt")
train_data1 = ["nrc_joy", "yelp_subset","amazon_finefood_subset","amazon_toys_subset"]
test_data1 = train_data1+["song_joy", "dialog_joy", "friends_joy", "emobank"]

nlp = spacy.load("/home/zwu49/ztwu/empathy_dictionary/fasttext")

for testDf in test_data1:
    testData = generateFastTextData_Spacy(testDf, nlp, textVariable = 'text')
    testDataset = Dataset(testDf, testData)

    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=8, 
                                            shuffle=False, num_workers=1)

    acc, f1 = testModel_FFN(model, testLoader)
    print(acc, f1)
