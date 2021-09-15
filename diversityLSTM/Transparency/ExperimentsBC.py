from Transparency.common_code.common import *
from Transparency.Trainers.PlottingBC import generate_graphs
from Transparency.configurations import configurations
from Transparency.Trainers.TrainerBC import Trainer, Evaluator, RationaleTrainer
from Transparency.Trainers.DatasetBC import *

def train_dataset(dataset, config='lstm') :

    config = configurations[config](dataset)
    trainer = Trainer(dataset, config=config, _type=dataset.trainer_type)
    if hasattr(dataset,'n_iter'):
        n_iters = dataset.n_iter
    else:
        n_iters = 8
    
    trainer.train(dataset.train_data, dataset.dev_data, n_iters=n_iters, save_on_metric=dataset.save_on_metric)
    # evaluator = Evaluator(dataset, trainer.model.dirname, _type=dataset.trainer_type)
    # _ = evaluator.evaluate(dataset.test_data, save_results=True)
    # return trainer, evaluator
    return trainer

def train_dataset_on_encoders(dataset, encoders, args=None) :
    for e in encoders :
        train_dataset(dataset, e)
        run_experiments_on_latest_model(dataset, e, args=args)
        # Yilin Geng
        #run_rationale_on_latest_model(dataset, e)
        # end
        
def generate_graphs_on_encoders(dataset, encoders) :
    for e in encoders :
        generate_graphs_on_latest_model(dataset, e)

def run_rationale_on_latest_model(dataset, config='lstm') :
    config = configurations[config](dataset)
    latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))
    rationale_gen = RationaleTrainer(dataset, config, latest_model, _type=dataset.trainer_type)
    print ('Training the Rationale Generator ...')
    _ = rationale_gen.train(dataset.train_data,dataset.dev_data)
    print ('Running Exp to Compute Attention given to Rationales ...')
    rationale_gen.rationale_attn_experiment(dataset.test_data)
    return rationale_gen

def run_evaluator_on_latest_model(dataset, config='lstm', args=None) :
    config = configurations[config](dataset)
    latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))
    evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)
    # _ = evaluator.evaluate(dataset.test_data, save_results=True)

    # #Yilin
    train_dataset_list = ['nrc_joy','nrc_anger','nrc_sadness','nrc_surprise','nrc_fear','amazon_toys_subset','amazon_finefood_subset','yelp_subset']
    # external_dataset_list = ['friends_joy','friends_anger','friends_sadness','friends_surprise','friends_fear','dialog_joy','dialog_anger','dialog_sadness','dialog_surprise','dialog_fear','song_joy','song_anger','song_sadness','song_surprise','song_fear','emobank']
    posneg_train_list = ['nrc_joy','amazon_toys_subset','amazon_finefood_subset','yelp_subset']
    posneg_eval_list = ['nrc_joy','friends_joy','song_joy','dialog_joy','emobank','amazon_toys_subset','amazon_finefood_subset','yelp_subset']
    if dataset.name in posneg_train_list:
        for eval_dataset_name in posneg_eval_list:
            eval_dataset = datasets[eval_dataset_name](args)
            if eval_dataset_name in train_dataset_list:
                _ = evaluator.evaluate(dataset.name, eval_dataset.name, eval_dataset.test_data, save_results=True)
            else:
                # for external dataset, all data are stored in the train set
                _ = evaluator.evaluate(dataset.name, eval_dataset.name, eval_dataset.train_data, save_results=True)
    else:
        emo = dataset.name.split('_',1)[1]
        for eval_dataset_name in ['nrc_', 'song_', 'dialog_', 'friends_']:
            eval_dataset = datasets[eval_dataset_name+emo](args)
            if eval_dataset_name in train_dataset_list:
                _ = evaluator.evaluate(dataset.name, eval_dataset.name, eval_dataset.test_data, save_results=True)
            else:
                _ = evaluator.evaluate(dataset.name, eval_dataset.name, eval_dataset.train_data, save_results=True)
    
    return evaluator

#Yilin Geng 
def run_experiments_on_latest_model(dataset, config='lstm', force_run=True, args=None) :
        evaluator = run_evaluator_on_latest_model(dataset, config, args)
        #Yilin
        train_data = dataset.train_data
        #evaluator.gradient_experiment(test_data, force_run=force_run)
        evaluator.quantitative_analysis_experiment(train_data, dataset, force_run=force_run)
        #evaluator.importance_ranking_experiment(test_data, force_run=force_run)
        #evaluator.conicity_analysis_experiment(test_data)
        #evaluator.permutation_experiment(test_data, force_run=force_run)
        #evaluator.integrated_gradient_experiment(dataset, force_run=force_run)
#end

def generate_graphs_on_latest_model(dataset, config='lstm'):

    config = configurations[config](dataset)
    latest_model = get_latest_model(os.path.join(config['training']['basepath'], config['training']['exp_dirname']))
    evaluator = Evaluator(dataset, latest_model, _type=dataset.trainer_type)
    _ = evaluator.evaluate(dataset.test_data, save_results=False)
    generate_graphs(dataset, config['training']['exp_dirname'], evaluator.model, test_data=dataset.test_data)
