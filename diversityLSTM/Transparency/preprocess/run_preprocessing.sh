export LSTM=diversityLSTM/Transparency

python $LSTM/preprocess/dataset_process.py --data_folder /Users/nil/Desktop/JHU/Word/data

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/nrc_joy_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_nrc_joy.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/nrc_anger_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_nrc_anger.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/nrc_sadness_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_nrc_sadness.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/nrc_surprise_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_nrc_surprise.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/nrc_fear_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_nrc_fear.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/empathy_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_empathy.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/amazon_toys_subset_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_amazon_toys_subset.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/amazon_finefood_subset_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_amazon_finefood_subset.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/yelp_subset_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_yelp_subset.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/emobank_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_emobank.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/friends_joy_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_friends_joy.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/friends_anger_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_friends_anger.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/friends_sadness_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_friends_sadness.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/friends_surprise_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_friends_surprise.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/friends_fear_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_friends_fear.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/song_joy_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_song_joy.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/song_anger_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_song_anger.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/song_sadness_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_song_sadness.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/song_surprise_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_song_surprise.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/song_fear_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_song_fear.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/dialog_joy_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_dialog_joy.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/dialog_anger_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_dialog_anger.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/dialog_sadness_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_dialog_sadness.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/dialog_surprise_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_dialog_surprise.p --min_df 0

python $LSTM/preprocess/preprocess_data_BC.py --data_file $LSTM/preprocess/ourData/dialog_fear_dataset.csv --tokenizer_file fasttext --output_file $LSTM/preprocess/ourData/vec_dialog_fear.p --min_df 0

