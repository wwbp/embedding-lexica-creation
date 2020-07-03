import argparse
import os
import logging

import torch

from scripts.run_bert import run_train
from scripts.predict_essay import run_predict


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", required=True, type=str, 
                    help="The input data dir. Should contain the .csv files"
                    "for the task.")
parser.add_argument("--task", required=True, type=str, 
                    help="The task for empathy or distress.")
parser.add_argument("--do_train", type=bool, default=False, help="Whether to run training.")
parser.add_argument("--do_predict", type=bool, default=False, 
                    help="Whether to run the model in inference mode on the test set.")
parser.add_argument("--k_fold", type=int, default=0, help="Whether to use k-fold validation.z")
parser.add_argument("--model", required=True, type=str, 
                    help="The pretrained Bert model we choose.")
parser.add_argument("--do_lower_case", type=bool, default=True,
                    help= "Whether to lower case the input text. Should be True for uncased "
                    "models and False for cased models.")
parser.add_argument("--max_seq_length", type=int, default=128,
                    help="The maximum total input sequence length after WordPiece tokenization. "
                    "Sequences longer than this will be truncated, and sequences shorter "
                    "than this will be padded.")
parser.add_argument("--train_batch_size", type=int, default=32, help="Total batch size for training.")
parser.add_argument("--eval_batch_size", type=int, default=8, help="Total batch size for eval.")
parser.add_argument("--predict_batch_size", type=int, default=8, help="Total batch size for prediction.")
parser.add_argument("--lr", type=float, default=1e-5, help="The initial learning rate for Adam.")
parser.add_argument("--epsilon", type=float, default=1e-8, help="Decay rate for Adam.")
parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
parser.add_argument("--num_warmup_steps", type=int, default=0, help="Steps of training to perform linear learning rate warmup for.")
parser.add_argument("--output_dir", type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--tokenizer", type=str, help="Dir to tokenizer for prediction.")
parser.add_argument("--early_stop", type=bool, default=False, help="Whether set early stopping based on F-score.")
parser.add_argument("--patience", type=int, default=7, help="patience for early stopping.")
parser.add_argument("--delta", type=float, default=0, help="delta for early stopping.")

args = parser.parse_args()
    

if __name__ == "__main__":

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    if args.do_train:
        run_train(device, args)
    elif args.do_predict:
        run_predict(device, args)
    else:
        Exception("Have to do one of the training or prediction!")
            
            
        
        
        
        
        
        
        
        