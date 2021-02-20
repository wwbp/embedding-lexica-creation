from Transparency.common_code.common import *
import Transparency.preprocess.vectorizer

def sortbylength(X, y) :
    len_t = np.argsort([len(x) for x in X])
    X1 = [X[i] for i in len_t]
    y1 = [y[i] for i in len_t]
    return X1, y1
    
def filterbylength(X, y, min_length = None, max_length = None) :
    lens = [len(x)-2 for x in X]
    min_l = min(lens) if min_length is None else min_length
    max_l = max(lens) if max_length is None else max_length

    idx = [i for i in range(len(X)) if len(X[i]) > min_l+2 and len(X[i]) < max_l+2]
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]

    return X, y

def set_balanced_pos_weight(dataset) :
    y = np.array(dataset.train_data.y)
    dataset.pos_weight = [len(y) / sum(y) - 1]

class DataHolder() :
    def __init__(self, X, y) :
        self.X = X
        self.y = y
        self.attributes = ['X', 'y']

    def get_stats(self, field) :
        assert field in self.attributes
        lens = [len(x) - 2 for x in getattr(self, field)]
        return {
            'min_length' : min(lens),
            'max_length' : max(lens),
            'mean_length' : np.mean(lens),
            'std_length' : np.std(lens)
        }
    
    def mock(self, n=200) :
        data_kwargs = { key: getattr(self, key)[:n] for key in self.attributes}
        return DataHolder(**data_kwargs)

    def filter(self, idxs) :
        data_kwargs = { key: [getattr(self, key)[i] for i in idxs] for key in self.attributes}
        return DataHolder(**data_kwargs)

class Dataset() :
    def __init__(self, name, path, min_length=None, max_length=None, args=None) :
        self.name = name
        if args is not None and hasattr(args, 'data_dir') :
            path = os.path.join(args.data_dir, path)
        out_of_domain = False
        if args is not None and hasattr(args, 'in_domain_dataset_name'):
            in_domain_vec = pickle.load(open(args.in_domain_dataset_name, 'rb'))
            out_of_domain = True

        self.vec = pickle.load(open(path, 'rb'))
        #Yilin Geng
        train_dataset_list = ['nrc_joy','nrc_anger','nrc_sadness','nrc_surprise','nrc_fear','empathy','amazon_toys_subset','amazon_finefood_subset','yelp_subset']
        if self.name in train_dataset_list:
            X, Xd, Xt = self.vec.seq_text['train'], self.vec.seq_text['dev'], self.vec.seq_text['test']
            y, yd, yt = self.vec.label['train'], self.vec.label['dev'], self.vec.label['test']

            if out_of_domain and args.in_domain_dataset_name != path:
                print("NOTE: converting indices from ", path, "to", args.in_domain_dataset_name)
                Xt = self.convert(in_domain_vec.word2idx, self.vec.map2words, Xt)


            X, y = filterbylength(X, y, min_length=min_length, max_length=max_length)
            Xt, yt = filterbylength(Xt, yt, min_length=min_length, max_length=max_length)
            Xt, yt = sortbylength(Xt, yt)


            Xd, yd = filterbylength(Xd, yd, min_length=min_length, max_length=max_length)
            Xd, yd = sortbylength(Xd, yd)

            self.train_data = DataHolder(X, y)
            self.dev_data = DataHolder(Xd, yd)
            self.test_data = DataHolder(Xt, yt)

        else:
            # Use train set to store evaluation data for external datasets including friends, dialog, song and emobank
            X = self.vec.seq_text['train']
            y = self.vec.label['train']

            X = self.convert(in_domain_vec.word2idx, self.vec.map2words, X)

            X, y = filterbylength(X, y, min_length=min_length, max_length=max_length)
            X, y = sortbylength(X, y)
            self.train_data = DataHolder(X, y)

        #end
        
        self.trainer_type = 'Single_Label'
        self.output_size = 1
        self.save_on_metric = 'roc_auc'
        self.keys_to_use = {
            'roc_auc' : 'roc_auc', 
            'pr_auc' : 'pr_auc'
        }

        self.bsize = 32
        if args is not None and hasattr(args, 'output_dir') :
            self.basepath = args.output_dir

    def safe_word2idx(self, word, word2idx):
        if word in word2idx:
            return word2idx[word]
        else:
            return word2idx['<UNK>']

    def convert(self, word2idx, map_to_words, X):
        for i,r in enumerate(X):
            #print(r)
            words = map_to_words(r)
            #print(words)
            X[i] = [self.safe_word2idx(w, word2idx) for w in words]
            #print(X[i])
        
            #import pdb; pdb.set_trace()

        return X

    def display_stats(self) :
        stats = {}
        stats['vocab_size'] = self.vec.vocab_size
        stats['embed_size'] = self.vec.word_dim
        y = np.unique(np.array(self.train_data.y), return_counts=True)
        yt = np.unique(np.array(self.test_data.y), return_counts=True)

        stats['train_size'] = list(zip(y[0].tolist(), y[1].tolist()))
        stats['test_size'] = list(zip(yt[0].tolist(), yt[1].tolist()))
        stats.update(self.train_data.get_stats('X'))

        outdir = "datastats"
        os.makedirs('graph_outputs/' + outdir, exist_ok=True)

        json.dump(stats, open('graph_outputs/' + outdir + '/' + self.name + '.txt', 'w'))
        print(stats)

########################################## Dataset Loaders ################################################################################

# def SST_dataset(args=None) :
#     dataset = Dataset(name='sst', path='diversityLSTM/Transparency/preprocess/SST/vec_sst.p', min_length=5, args=args)
#     set_balanced_pos_weight(dataset)
#     return dataset

# def IMDB_dataset(args=None) :
#     dataset = Dataset(name='imdb', path='diversityLSTM/Transparency/preprocess/IMDB/vec_imdb.p', min_length=6, args=args)
#     set_balanced_pos_weight(dataset)
#     return dataset

# def News20_dataset(args=None) :
#     dataset = Dataset(name='20News_sports', path='diversityLSTM/Transparency/preprocess/20News/vec_20news_sports.p', min_length=6, max_length=500, args=args)
#     set_balanced_pos_weight(dataset)
#     return dataset

# def Yelp(args=None) :
#     dataset = Dataset(name='Yelp', path='diversityLSTM/Transparency/preprocess/Yelp/vec_yelp.p', min_length=6, max_length = 150, args=args)
#     set_balanced_pos_weight(dataset)
#     return dataset

# def Amazon(args=None) :
#     dataset = Dataset(name='Amazon', path='diversityLSTM/Transparency/preprocess/Amazon/vec_amazon.p', min_length=6, max_length=100, args=args)
#     set_balanced_pos_weight(dataset)
#     return dataset

# def ADR_dataset(args=None) :
#     dataset = Dataset(name='tweet', path='diversityLSTM/Transparency/preprocess/Tweets/vec_adr.p', min_length=5, max_length=100, args=args)
#     set_balanced_pos_weight(dataset)
#     return dataset

# def Anemia_dataset(args=None) :
#     dataset = Dataset(name='anemia', path='diversityLSTM/Transparency/preprocess/MIMIC/vec_anemia.p', max_length=4000, args=args)
#     set_balanced_pos_weight(dataset)
#     return dataset

# def Diabetes_dataset(args=None) :
#     dataset = Dataset(name='diab', path='diversityLSTM/Transparency/preprocess/MIMIC/vec_diabetes.p', min_length=6, max_length=4000, args=args)
#     set_balanced_pos_weight(dataset)
#     return dataset

#Yilin Geng 09/07/2020
def nrc_joy_dataset(args=None):
    dataset = Dataset(name='nrc_joy', path='diversityLSTM/Transparency/preprocess/ourData/vec_nrc_joy.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def nrc_sadness_dataset(args=None):
    dataset = Dataset(name='nrc_sadness', path='diversityLSTM/Transparency/preprocess/ourData/vec_nrc_sadness.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def nrc_fear_dataset(args=None):
    dataset = Dataset(name='nrc_fear', path='diversityLSTM/Transparency/preprocess/ourData/vec_nrc_fear.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def nrc_anger_dataset(args=None):
    dataset = Dataset(name='nrc_anger', path='diversityLSTM/Transparency/preprocess/ourData/vec_nrc_anger.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def nrc_surprise_dataset(args=None):
    dataset = Dataset(name='nrc_surprise', path='diversityLSTM/Transparency/preprocess/ourData/vec_nrc_surprise.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def empathy_dataset(args=None):
    dataset = Dataset(name='empathy', path='diversityLSTM/Transparency/preprocess/ourData/vec_empathy.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def amazon_finefood_subset_dataset(args=None):
    dataset = Dataset(name='amazon_finefood_subset', path='diversityLSTM/Transparency/preprocess/ourData/vec_amazon_finefood_subset.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def amazon_toys_subset_dataset(args=None):
    dataset = Dataset(name='amazon_toys_subset', path='diversityLSTM/Transparency/preprocess/ourData/vec_amazon_toys_subset.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def yelp_subset_dataset(args=None):
    dataset = Dataset(name='yelp_subset', path='diversityLSTM/Transparency/preprocess/ourData/vec_yelp_subset.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

#External datasets
def emobank_dataset(args=None) :
    dataset = Dataset(name='emobank', path='diversityLSTM/Transparency/preprocess/ourData/vec_emobank.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def song_joy_dataset(args=None) :
    dataset = Dataset(name='song_joy', path='diversityLSTM/Transparency/preprocess/ourData/vec_song_joy.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def song_sadness_dataset(args=None) :
    dataset = Dataset(name='song_sadness', path='diversityLSTM/Transparency/preprocess/ourData/vec_song_sadness.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def song_fear_dataset(args=None) :
    dataset = Dataset(name='song_fear', path='diversityLSTM/Transparency/preprocess/ourData/vec_song_fear.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def song_anger_dataset(args=None) :
    dataset = Dataset(name='song_anger', path='diversityLSTM/Transparency/preprocess/ourData/vec_song_anger.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def song_surprise_dataset(args=None) :
    dataset = Dataset(name='song_surprise', path='diversityLSTM/Transparency/preprocess/ourData/vec_song_surprise.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def dialog_joy_dataset(args=None) :
    dataset = Dataset(name='dialog_joy', path='diversityLSTM/Transparency/preprocess/ourData/vec_dialog_joy.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def dialog_sadness_dataset(args=None) :
    dataset = Dataset(name='dialog_sadness', path='diversityLSTM/Transparency/preprocess/ourData/vec_dialog_sadness.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def dialog_fear_dataset(args=None) :
    dataset = Dataset(name='dialog_fear', path='diversityLSTM/Transparency/preprocess/ourData/vec_dialog_fear.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def dialog_anger_dataset(args=None) :
    dataset = Dataset(name='dialog_anger', path='diversityLSTM/Transparency/preprocess/ourData/vec_dialog_anger.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def dialog_surprise_dataset(args=None) :
    dataset = Dataset(name='dialog_surprise', path='diversityLSTM/Transparency/preprocess/ourData/vec_dialog_surprise.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

def friends_joy_dataset(args=None) :
    dataset = Dataset(name='friends_joy', path='diversityLSTM/Transparency/preprocess/ourData/vec_friends_joy.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def friends_sadness_dataset(args=None) :
    dataset = Dataset(name='friends_sadness', path='diversityLSTM/Transparency/preprocess/ourData/vec_friends_sadness.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def friends_fear_dataset(args=None) :
    dataset = Dataset(name='friends_fear', path='diversityLSTM/Transparency/preprocess/ourData/vec_friends_fear.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def friends_anger_dataset(args=None) :
    dataset = Dataset(name='friends_anger', path='diversityLSTM/Transparency/preprocess/ourData/vec_friends_anger.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset
def friends_surprise_dataset(args=None) :
    dataset = Dataset(name='friends_surprise', path='diversityLSTM/Transparency/preprocess/ourData/vec_friends_surprise.p', min_length=6, args=args)
    set_balanced_pos_weight(dataset)
    return dataset

# def amazon_baby_dataset(args=None) :
#     dataset = Dataset(name='amazon_baby_subset', path='diversityLSTM/Transparency/preprocess/ourData/vec_amazon_baby_subset.p', min_length=6, args=args)
#     set_balanced_pos_weight(dataset)
#     return dataset
# def nyelp_dataset(args=None) :
#     dataset = Dataset(name='nyelp', path='diversityLSTM/Transparency/preprocess/ourData/vec_nyelp.p', min_length=6, args=args)
#     set_balanced_pos_weight(dataset)
#     return dataset
# def nyelp_1v5_dataset(args=None) :
#     dataset = Dataset(name='nyelp_1v5', path='diversityLSTM/Transparency/preprocess/ourData/vec_nyelp_1v5.p', min_length=6, args=args)
#     set_balanced_pos_weight(dataset)
#     return dataset
#end

datasets = {
    # "sst" : SST_dataset,
    # "imdb" : IMDB_dataset,
    # 'amazon': Amazon,
    # 'yelp': Yelp,
    # "20News_sports" : News20_dataset,
    # "tweet" : ADR_dataset ,
    # "Anemia" : Anemia_dataset,
    # "Diabetes" : Diabetes_dataset,
    #Yilin Geng 
    # "nyelp" : nyelp_dataset,
    # "nyelp_1v5" : nyelp_1v5_dataset,
    "yelp_subset" : yelp_subset_dataset,
    "amazon_finefood_subset" : amazon_finefood_subset_dataset,
    "amazon_toys_subset" : amazon_toys_subset_dataset,
    # "amazon_baby" : amazon_baby_dataset,
    "empathy" : empathy_dataset,
    "nrc_joy" : nrc_joy_dataset,
    "nrc_sadness" : nrc_sadness_dataset,
    "nrc_fear" : nrc_fear_dataset,
    "nrc_anger" : nrc_anger_dataset,
    "nrc_surprise" : nrc_surprise_dataset,
    "emobank" : emobank_dataset,
    "song_joy" : song_joy_dataset,
    "song_sadness" : song_sadness_dataset,
    "song_fear" : song_fear_dataset,
    "song_anger" : song_anger_dataset,
    "song_surprise" : song_surprise_dataset,
    "dialog_joy" : dialog_joy_dataset,
    "dialog_sadness" : dialog_sadness_dataset,
    "dialog_fear" : dialog_fear_dataset,
    "dialog_anger" : dialog_anger_dataset,
    "dialog_surprise" : dialog_surprise_dataset,
    "friends_joy" : friends_joy_dataset,
    "friends_sadness" : friends_sadness_dataset,
    "friends_fear" : friends_fear_dataset,
    "friends_anger" : friends_anger_dataset,
    "friends_surprise" : friends_surprise_dataset
    #end
}

