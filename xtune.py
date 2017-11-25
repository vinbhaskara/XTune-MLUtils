'''
Created on Nov 08, 2017

@author: Vineeth_Bhaskara
'''

from __future__ import print_function
import os, pickle
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
import numpy as np
from numba import jit # Compile intensive functions inline to C code for faster perf
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss
import sys, gc
import pandas as pd
import matplotlib.pyplot as plt 
import lightgbm as lgb
from mltools import *
'''
TODOS -
Caliberating the predictions
Bayesian optimization of the parameters (Check https://www.kaggle.com/tilii7/bayesian-optimization-of-xgboost-parameters)
MP support across param grid (new branch created)
Map os.listdir() to sorted(os.listdir())
Modify xPredict. Check if best_iteration of lgb or best_ntree_limit are neither 0 or 1 and then use them without asking (give option to specify numtrees though but default only till best).
'''

class Logger(object):
    '''
    Custom logger to log the print statements into a file to track processes for instance after closing Jupyter Screen
    '''
    
    def __init__(self, logfile='logfile'):
        self.terminal =  sys.__stdout__
        self.logfile = logfile
        

    def write(self, message):
        self.terminal.write(message)
        self.log = open(self.logfile+".log", "a")
        self.log.write(message) 
        self.log.close()
        sys.stdout.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass 


@jit
def eval_gini(y_true, y_prob):
    '''
    Normalized Gini Coefficient Measure -- somewhat related to the AUC -- but more related to the ordering of the predictions.
    Used in Kaggle for instance in the Safe Driver Prediction Challenge (binary classification pure xgboost competition).
    Implementation by CPMP.
    '''
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

@jit
def multiclass_log_loss(actual, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss
    idea from this post:
    http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/12209#post12209
    Parameters
    ----------
    y_true : array, shape = [n_samples]
    y_pred : array, shape = [n_samples, n_classes]
    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]
    rows = actual.shape[0]

    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota

def xgb_gini(pred, d_eval): 
    # more is better like auc; only for binary problems

    obs = d_eval.get_label()
    obs_onehot=[]

    if pred.shape[1] == 1:
        obs_onehot = obs
    elif pred.shape[1] == 2:            
        for i in obs:
            if i==0:
                obs_onehot.append([1, 0])
            else:
                obs_onehot.append([0, 1])
    else:
        print('Not valid for non-binary problems.')
        raise
    
    gini_score = eval_gini(obs, pred[:,1])

    return [('kaglloss', multiclass_log_loss(np.array(obs_onehot).astype(float), pred)), ('gini', gini_score), ('auc', roc_auc_score(np.array(obs_onehot).astype(float), pred))]

def xgb_auc(pred, d_eval):
    '''
    Sklearn auc for Xtune. 
    For binary class problems optimized with mlogloss for multi:softprob, this may be used for 
    as eval_metric for early_stopping, for example.

    Usage:
        1) pred: numpy matrix representing the binary class predictions 
        2) d_eval: xgb DMatrix of the data whose predictions are pred (above) - making use of the labels by .get_label() 
    '''
    obs = d_eval.get_label()
    obs_onehot=[]

    if pred.shape[1] == 1:
        obs_onehot = obs
    elif pred.shape[1] == 2:            
        for i in obs:
            if i==0:
                obs_onehot.append([1, 0])
            else:
                obs_onehot.append([0, 1])
    else:
        print('Not valid for non-binary problems.')
        raise

    return [('kaglloss', multiclass_log_loss(np.array(obs_onehot).astype(float), pred)), ('auc', roc_auc_score(np.array(obs_onehot).astype(float), pred))]
    
    
def xPredict( model, d_pred, boosting_alg='xgb', lgb_best_iteration=-1):
    '''
    Simple xgb/lgb Predict alternative with best_iteration implementation
    to avoid silly mistakes.
    
    when lgb_best_iteration is -1 that means all trees. (this parameter exists because unlike
    xgboost, lgb doesnt set model.best_iteration unless it is stopped by early stopping.
    '''
    
    if boosting_alg=='xgb':
        try:
            return model.predict(d_pred, ntree_limit=model.best_ntree_limit) # some problems occurred before with gblinear booster
        except:
            print('Getting error on trying ntree_limit. Proceeding with all trees.')
            return model.predict(d_pred)
        
    elif boosting_alg=='lgb':
        return model.predict(d_pred, num_iteration=lgb_best_iteration)

    else:
        print('Boosting should be either xgb or lgb. No valid option passed.')
        raise

def gcRefresh():
    gc.collect()


def xTrain( d_train, param, val_data=None, prev_model=None, verbose_eval=True, boosting_alg='xgb', 
           logfile=None,  lgb_categorical_feats='auto', lgb_learning_rates=None):
    '''
    Usage:
        1) d_train: xgb/lgb DMatrix of Train data (if lgb, preferably use free_raw_data=False while constructing,
        example: 
        d_train = lgb.Dataset(train_csv[cols], label=train_csv['target'], feature_name=cols, free_raw_data=False))
        
        2) param: Training parameters
               Example default param -
                param_xgb_default={'booster':'gbtree',
                                'silent':0, 
                                'num_estimators':1000,
                                'early_stopping':5,
                                'eval_metric':'mlogloss', # if feval is set then that overrides eval_metric
                                'objective':'multi:softprob',
                                'num_class':2,
                                'feval':'xgb_auc', # feval overrides eval_metric for early stopping. You may pass custom functions too.
                                'maximize_feval': True,
                                'tree_method':'exact' #'hist' for lightGBM-like
                                }
                
                lgbparam = { 'num_estimators': 250,
                             'is_unbalance': False,  
                             'num_leaves': 800, # number of leaves in one tree
                             'early_stopping': 5, 
                             'learning_rate': 0.1, 
                             'min_data_in_leaf': 25, 
                             'use_missing': False, # -1
                             'num_threads': 48, 
                             'objective': 'binary', 
                             'feature_fraction': 0.8, 
                             'predict_raw_score': True, 
                             'bagging_freq': 5, 
                             'bagging_fraction': 0.8, 
                             'lambda_l2': 0.0,
                             'feval':None,
                             'metric':'auc'}
        3) val_data: xgb DMatrix for validation data 
        4) prev_model: for continuing training
        5) verbose_eval: True/False - To display individual boosting_alg rounds stats
        
        Specify a list of Categorical columns in lgb_categorical_feats for using
        inbuilt OHE of LightGBM (valid only when boosting_alg='lgb')
        Note that the categorical columns MUST be of int type. If symbolic strings, 
        please convert to some numeric encoding before passing here.
        
        lgb_learning_rates: Check learning_rates option of lgb.train()
        
        Returns: trained model, and history dictionary
    '''
    if param is None:
        print('No param passed. Check an example: ', xtrain.__doc__)
        sys.exit()
        
    if logfile is not None:
        stdout_backup = sys.stdout
        sys.stdout = Logger(logfile)

    param_xgb = param.copy() 

    if param_xgb['feval'] == 'xgb_auc':
        param_xgb['feval'] = xgb_auc
    if param_xgb['feval'] == 'lgb_auc':
        param_xgb['feval'] = lgb_auc

    if 'num_estimators' not in list(param_xgb.keys()):
        print('Choosing default num_estimators: ', 5000)
        param_xgb['num_estimators'] = 5000
    if 'early_stopping' not in list(param_xgb.keys()):
        param_xgb['early_stopping'] = 5
    if 'feval' not in list(param_xgb.keys()):
        param_xgb['feval'] = None
        
    if 'boosting_alg' in list(param_xgb.keys()):
        boosting_alg = param_xgb['boosting_alg']
        
    
    if val_data is None:
        valid_sets=[d_train]
        valid_names=['train']
        watchlist=[(d_train, 'train')]
        if param_xgb['early_stopping'] is not None:
            print('Ignoring early stopping as no validation data passed.')
            param_xgb['early_stopping']=None
    else:
        valid_sets=[d_train, val_data]
        valid_names=['train','val']
        watchlist=[(d_train, 'train'),(val_data, 'val')]

    history_dict={}
    feval=None
    if param_xgb['feval'] is not None:
        feval=param_xgb['feval']
 
    else:
        if boosting_alg=='xgb':
            param['maximize_feval'] = False
            
    if 'feval' in param_xgb.keys():
        del param_xgb['feval']
        
    if boosting_alg=='xgb':
        
        model=xgb.train(param_xgb, d_train, num_boost_round=param_xgb['num_estimators'], 
                 evals=watchlist,feval=feval, maximize=param_xgb['maximize_feval'], # custom metric for early stopping
                 early_stopping_rounds=param_xgb['early_stopping'], 
                 evals_result=history_dict,
                 xgb_model=prev_model, # allows continuation of previously trained model
                 verbose_eval=verbose_eval)
        
    elif boosting_alg=='lgb':
        model=lgb.train(param_xgb, d_train, num_boost_round=param_xgb['num_estimators'],
                       valid_sets=valid_sets, valid_names=valid_names, feval=feval,
                       init_model=prev_model, categorical_feature=lgb_categorical_feats,
                       early_stopping_rounds=param_xgb['early_stopping'],
                       evals_result=history_dict, verbose_eval=verbose_eval,
                       learning_rates=lgb_learning_rates)
    if logfile is not None:        
        sys.stdout=stdout_backup
    return model, history_dict.copy()


def xGridSearch( d_train, params, lgb_raw_train=None, randomized=False, num_iter=None, rand_state=28081994, isCV=True, 
              folds=5, d_holdout=None, verbose_eval=True, save_models=False, skip_param_if_same_eval=False, save_prefix='',save_folder='./model_pool', limit_complexity=None, logfile=None, boosting_alg='xgb'):
    '''       

    Usage:
        1) d_train: xgb/lgb DMatrix of Train data (if lgb, preferably use free_raw_data=False while constructing,
        and while passing categorical columns, ensure the pandas dataframe passed to Dataset has those dtypes as categorical (then no need of onehotting)
        Example:
        
        for i in cols:
            if i in catcols:
                train_csv[i] = train_csv[i].astype('category')
            else:
                train_csv[i] = train_csv[i].astype(float)
        
        d_train = lgb.Dataset(train_csv[cols], label=train_csv['target'], feature_name=cols, free_raw_data=False)
        
        NOTE: If using LGB, you MUST set lgb_raw_train NUMPY MATRIX with RAW traindata that was used to make 
        lgb Dataset object.
        
        2) params: dictionary with list of possible values for a parameter as keys
                   Example params:
                    params_xgb_default_dict={
                        'booster':['gbtree', 'gblinear'],
                        'tree_method':['hist','exact'],
                        'silent':[0], 
                        'num_estimators':[1000],
                        'early_stopping':[5],
                        'eval_metric':['mlogloss'], # if feval is set then that overrides eval_metric
                        'objective':['multi:softprob'],
                        'eta': [0.1],
                        'max_depth':[12],
                        'min_child_weight':[0.05],
                        'gamma':[0.1],
                        'alpha':[0.1],
                        'lambda':[1e-05],
                        'subsample':[0.8],
                        'colsample_bytree':[0.8],
                        'num_class':[2, 4, 6, 8, 10, 12],
                        'feval':['xgb_auc'], # feval overrides eval_metric for early stopping. You may pass custom functions too.
                        'maximize_feval': [True],
                        'tree_method':['exact', 'hist']
                        }
                        
                    lgbparam = {'boosting_type':['gbdt'],
                     'num_estimators': [5000],
                     'is_unbalance': [False],  
                     'num_leaves': [8, 4, 12, 16], # number of leaves in one tree
                     'early_stopping': [20], 
                     'learning_rate': [0.05], 
                     'min_data_in_leaf': [25, 500], # min_child_samples
                     'use_missing': [True], # -1
                     'num_threads': [3], 
                     'objective': ['binary'],
                     'feature_fraction': [0.8, 0.3], # colsample_bytree
                     'predict_raw_score': [False], 
                     'bagging_freq': [5, 10, 2], # subsample_freq
                     'bagging_fraction': [0.8, 0.7], # subsample
                     'lambda_l2': [0.0],
                     'max_bin':[255, 10, 20], # def 255
                     'feval':[None],
                     'maximize_feval':[True],
                     'num_class':[1],
                     'max_depth':[-1, 4, 6], # def -1
                     'min_hessian':[1e-3, 0.05], # min_child_weight def 1e-3
                     'eval_metric':[['binary_logloss', 'auc']]}
                        
                       
        3) randomized: False/True - To randomly choose points from the parameter Grid for Search (without replacement)
        4) num_iter: Specified when randomized=True to limit the total number of random parameters considered. If None,
        run continues till the parameter grid is exhausted
        5) rand_state: for reproducibility
        6) isCV: True/False - Cross Validation OR Holdout
        7) folds: If isCV, then the number of CV folds, else represents the holdout portion to be 1/folds fraction of the
        training data provided given d_holdout is None
        8) d_holdout: Data for holdout. If specified, then folds has no effect.
        9) verbose_eval: True/False - verbosity - printing each round stats
        10) save_models: Save each and every model - both across CV and across param grid - For say like Stacking later on. Also saves val preds or holdout preds.
        11) save_prefix: prefix filename while saving model files
        12) save_folder: Folder where to save the models if save_models is True
        13) limit_complexity: Very useful function when trying to find the best model in a hyperparameter search with less number of rounds for fast decisions. Complexity is defined as max_depth*num_estimators. If limit_complexity is provided, then the num_estimators will be determined from the max_depth by num_estimators=limit_complexity/max_depth so that the hyperparam searching is fair. (Eg. Think of optimizing max_depth=[1,2] with 5 rounds. Obvly, that is unfair, as the later is more complex than former).
        14) skip_param_if_same_eval: This option saves the model while iterating only if the eval metric value for that parameter results in a number that has already not resulted previously (eg. some iterations over regularizations alone produce the exact same result). In case of CV folds, all 1st folds' eval is maintained, and if current matches that, remaining rounds are skipped saving time. (Also useful while ensembling a population of models for Kaggle).
        15) logfile: specify a logfile to also print to file in addition to stdout, for example, for logging status even while Jupyter screen is closed.
    Note 1:
        If isCV is True does Cross Validation (Stratified) for folds times over d_train data.
        If isCV is False, then does a holdout by taking the d_holdout data.
        If isCV is False and d_holdout is None, then one stratified split of (100-100/folds):100/folds 
        is made train/holdout, and returns the holdout indices.
        
    Note 2:
        Last member of the eval_metric is used for select best param during CV/GridSearch if feval is None.


    Return Value: A dictionary with the following keys:
        best_param, best_eval, best_eval_folds, all_param_scores, *best_model, best_ntree_limit,
        holdout indices, train indices, best_validation_predictions, best_plots_lst

    *best_model would be the a list of models across folds of the best parameter.

    '''
    if boosting_alg=='lgb' and lgb_raw_train is None:
        print('Please set lgb_raw_train before continuing.')
        sys.exit()
    if logfile is not None:
        stdout_backup=sys.stdout
        sys.stdout=Logger(logfile)
    
    best_param=None
    best_eval=None
    all_param_scores=[]
    holdout_indices=[]
    train_indices=list(range(int(len(d_train.get_label().tolist()))))
    best_ntree_limit=None
    best_validation_predictions=None
    best_model=None
    best_param_scores=None
    best_cv_fold=None
    best_eval_folds=None
    
    if save_folder is not None:
        os.system('mkdir -p '+save_folder)
        os.system('mkdir -p '+save_folder+'/valpred')
        os.system('mkdir -p '+save_folder+'/model')
        os.system('mkdir -p '+save_folder+'/history')
        os.system('mkdir -p '+save_folder+'/param')

    if not isCV and d_holdout is None:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rand_state)
        print('Making a ', 100-100//folds, ' and ', 100//folds, ' split of Train:Test for Holdout.')
        for tr, ts in skf.split(np.zeros(len(d_train.get_label().tolist())), d_train.get_label()):
            
            if boosting_alg=='xgb':
                d_holdout = d_train.slice(ts)
                d_train = d_train.slice(tr)
            elif boosting_alg=='lgb':
                d_holdout = d_train.subset(ts)
                d_train = d_train.subset(tr)
                x_holdout = lgb_raw_train[ts]
            
            holdout_indices=ts
            train_indices=tr
            break       

    if rand_state is not None:
        np.random.seed(rand_state)

    pg = ParameterGrid(params)
    pglen=len(pg)
    print('Total Raw Grid Search Space to Sample: ', pglen)
    if num_iter is None:
        num_iter=len(pg)
    if randomized:
        indices = np.random.choice(range(0, pglen), size=num_iter, replace=False)            
        print(type(indices), type(pg))
        allparams = np.array(list(pg))[indices]
    else:
        allparams = pg

    total = len(allparams)
    counter=0
    
    model_first_fold_eval = [] # maintaing list of eval metric to skip repeat evals (check skip_param_if_same_eval)
    
    for param in allparams:
        counter+=1

        skipParam=False
        best_ntree_limit_folds=[]
        best_ntree_score_folds=[]
        ntree_hist_scores_folds=[]
        best_model_lst=[]

        val_pred=None # validation predictions of this param

        print('\n')
        print('#######################################################################')

        is_eval_more_better = False # error metric assumed default
        if param['feval'] is not None and param['maximize_feval'] is None:
            print('If want to use feval you must set maximize_feval for the grid search to know to keep best. Now continuing without feval.')
        elif param['feval'] is not None and param['maximize_feval'] == True:
            is_eval_more_better = True
            print('The eval metric is being maximized.')               
        

        if param['feval'] is None and (param['eval_metric']=='auc' or param['eval_metric'][-1]=='auc'):
            is_eval_more_better = True
            print('The eval metric is being maximized.')
            
        if boosting_alg=='lgb': # Lgb doesnt take this param
            if 'maximize_feval' in param.keys():
                del param['maximize_feval']
                
            if 'eval_metric' in param.keys():
                param['metric'] = param['eval_metric']
                del param['eval_metric']
            

        if not is_eval_more_better:
            print('The eval metric is being minimized.')
            
        if limit_complexity is not None:
            
            if boosting_alg=='xgb':
                num_estimators = int(int(limit_complexity)/param['max_depth']) # depth wise complexity per round
            elif boosting_alg=='lgb':
                num_estimators = int(int(limit_complexity)/param['num_leaves']) # leaves wise complexity per round
            if num_estimators<=0:
                print('Limit estimators passed, but this round has resultant num_estimators <=0. Hence skipping.')
                continue
                
            param['num_estimators']=num_estimators
            print('num estimators under limit complexity: ', num_estimators)


        print('Doing param ', counter, ' of total ', total,' - ', param, '\n')
        if not isCV: # holdout set 

            model, hist = xTrain(d_train, param, d_holdout, verbose_eval=verbose_eval, logfile=logfile, boosting_alg=boosting_alg)
            
            
            if boosting_alg=='xgb':
                val_pred = xPredict(model, d_holdout, boosting_alg)
                
                now_best_score = model.best_score #xgb
                now_best_limit = model.best_ntree_limit
            elif boosting_alg=='lgb':
                
                metric_to_use = param['metric']
                
                if str(type(metric_to_use)) == "<type 'list'>":
                    metric_to_use=param['metric'][-1]
                    
                if is_eval_more_better:
                    now_best_score = max(hist['val'][metric_to_use])
                else:
                    now_best_score = min(hist['val'][metric_to_use])

                now_best_limit = hist['val'][metric_to_use].index(now_best_score) + 1
                val_pred = xPredict(model, x_holdout, boosting_alg, lgb_best_iteration=now_best_limit)
                
            if skip_param_if_same_eval:
                if now_best_score in model_first_fold_eval:
                    print('Not doing param as same eval already acheived.')
                    skipParam=True
                    continue
                else:
                    model_first_fold_eval.append(now_best_score)
            
            if save_models:                        
                        
                filename= save_prefix+'_holdout_'+'param'+str(counter)
                fmodel = open(save_folder+'/model/'+filename+'.model', 'wb')
                pickle.dump(model, fmodel)
                fmodel.close()

                fhist = open(save_folder+'/history/'+filename+'.hist', 'wb')
                pickle.dump(hist, fhist)
                fhist.close()
                
                fparam = open(save_folder+'/param/'+filename+'.param', 'wb')
                pickle.dump(param, fparam)
                fparam.close()
                
                valpreddf = pd.DataFrame(val_pred)
                valpreddf.to_csv(save_folder+'/valpred/'+filename+'.holdout')

            print('Holdout: Score ', now_best_score, ' Trees ',now_best_limit)
            print('\n')

            best_ntree_limit_folds.append(now_best_limit)
            best_ntree_score_folds.append(now_best_score)
            ntree_hist_scores_folds.append(hist)              
            best_model_lst.append(model)


        else: # cross-validation     

            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rand_state)
            skf_split = skf.split(np.zeros(len(d_train.get_label().tolist())), d_train.get_label())
            foldcounter=0
            for tr, ts in skf_split:
                foldcounter+=1
                print('Doing CV fold #', foldcounter)
                
                if boosting_alg=='xgb':
                    xgb_train_cv = d_train.slice(tr)
                    xgb_val_cv = d_train.slice(ts)
                elif boosting_alg=='lgb':
                    xgb_train_cv = d_train.subset(tr)
                    xgb_val_cv = d_train.subset(ts)
                    x_val_cv = lgb_raw_train[ts]
                
                if boosting_alg=='xgb':
                    # xgboost seems to miss the feature labels after slicing
                    xgb_train_cv.feature_names=d_train.feature_names
                    xgb_val_cv.feature_names=d_train.feature_names
                
                model, hist = xTrain(xgb_train_cv, param, xgb_val_cv, verbose_eval=verbose_eval, logfile=logfile, boosting_alg=boosting_alg)

                if boosting_alg=='xgb':
                    val_pred_fold = xPredict(model, xgb_val_cv, boosting_alg)
                    
                    now_best_score = model.best_score #xgb
                    now_best_limit = model.best_ntree_limit
                elif boosting_alg=='lgb':
                   
                    metric_to_use = param['metric']

                    if str(type(metric_to_use)) == "<type 'list'>":
                        metric_to_use=param['metric'][-1]
                        
                    if is_eval_more_better:
                        now_best_score = max(hist['val'][metric_to_use])
                    else:
                        now_best_score = min(hist['val'][metric_to_use])

                    now_best_limit = hist['val'][metric_to_use].index(now_best_score) + 1                    
                    val_pred_fold = xPredict(model, x_val_cv, boosting_alg, lgb_best_iteration=now_best_limit)
                    
               
                if foldcounter == 1:
                    if skip_param_if_same_eval:
                        if now_best_score in model_first_fold_eval:
                            print('Not doing param as same eval already acheived.')
                            skipParam=True
                            break
                        else:
                            model_first_fold_eval.append(now_best_score)
                
                if save_models:
                    filename=save_prefix+'_cv_'+'param'+str(counter)+'_fold'+str(foldcounter)
                    fmodel = open(save_folder+'/model/'+filename+'.model', 'wb')
                    pickle.dump(model, fmodel)
                    fmodel.close()
                    
                    fhist = open(save_folder+'/history/'+filename+'.hist', 'wb')
                    pickle.dump(hist, fhist)
                    fhist.close()
                                    
                if val_pred is None:
                    if len(val_pred_fold.shape)==1: # for binary cases
                        val_pred=np.zeros((int(len(d_train.get_label().tolist())),))
                    else:
                        val_pred=np.zeros((int(len(d_train.get_label().tolist())), val_pred_fold.shape[1]))
                val_pred[ts]=val_pred_fold

                print('CV Fold: Score ', now_best_score, ' Trees ',now_best_limit)
                print('\n')

                best_ntree_limit_folds.append(now_best_limit)
                best_ntree_score_folds.append(now_best_score)
                ntree_hist_scores_folds.append(hist)
                best_model_lst.append(model)
                
            if skipParam:
                continue
                
        if save_models:
            fname = save_prefix+'_cv_'+'param'+str(counter)
            valpreddf = pd.DataFrame(val_pred)
            valpreddf.to_csv(save_folder+'/valpred/'+fname+'.validation')
            
            fparam = open(save_folder+'/param/'+fname+'.param', 'wb')
            pickle.dump(param, fparam)
            fparam.close()

        if is_eval_more_better:
            best_score_across_folds=max(best_ntree_score_folds)
            best_fold = best_ntree_score_folds.index(max(best_ntree_score_folds))+1
        else:
            best_score_across_folds=min(best_ntree_score_folds)
            best_fold = best_ntree_score_folds.index(min(best_ntree_score_folds))+1
        best_ntree_limit_across_folds=best_ntree_limit_folds[best_fold-1]
        
        current_eval = sum(best_ntree_score_folds)/float(len(best_ntree_score_folds)) # Average Eval Metric
        stddev_eval = np.std(np.array(best_ntree_score_folds))

        all_param_scores.append([param.copy(), {'ntree_limit_folds': best_ntree_limit_folds, \
                                                'best_ntree_score_folds': best_ntree_score_folds, \
                                                'ntree_hist_scores_folds': ntree_hist_scores_folds, \
                                                'score_avg': sum(best_ntree_score_folds)/float(len(best_ntree_score_folds)), \
                                                'best_ntree_limit_across_folds': best_ntree_limit_across_folds, \
                                                'best_score_across_folds': best_score_across_folds, \
                                                'best_fold': best_fold,
                                                'avg_cv_score':current_eval,
                                                'stddev_cv_score':stddev_eval}])
        
        if save_models:
            fname = save_prefix+'_'+'param'+str(counter)
            valpreddf = pd.DataFrame(val_pred)
            valpreddf.to_csv(save_folder+'/param/'+fname+'.paramstats')
        

        update=False
        if best_eval is None:
            update=True
        else:
            if is_eval_more_better:
                if current_eval > best_eval:
                    update=True
            else:
                if current_eval < best_eval:
                    update=True

        if update:
            best_eval_stddev=stddev_eval
            best_eval = current_eval
            best_param = param.copy()
            best_ntree_limit = best_ntree_limit_across_folds
            best_validation_predictions=val_pred
            best_model = best_model_lst
            best_param_scores=all_param_scores[-1]
            best_cv_fold=best_fold
            best_eval_folds=best_ntree_score_folds

        print('Params: ',param, '\nCV Rounds: ', best_ntree_limit_folds, '\nCV Scores: ', best_ntree_score_folds, ' \nAvg CV Score: ', sum(best_ntree_score_folds)/float(len(best_ntree_score_folds)), \
        '\nStdDev CV score: ',stddev_eval,'\nBest Fold: ', best_fold, '\nNumTreesForBestFold: ', best_ntree_limit_across_folds)

    print('\n')
    print('***********************************************************************')
    print('Final Results\n')
    print('Best Params: ',best_param, '\nCV Scores: ', best_eval_folds, ' \nAvg CV Score: ', best_eval, '\nStdDev CV score: ',best_eval_stddev,'\nBest Fold: ', \
best_cv_fold, '\nNumTreesForBestFold: ', best_ntree_limit)


    print('''\n\nReturned values are: best_model, best_eval_folds, best_cv_fold, best_ntree_limit, best_param, best_eval, best_param_scores, best_validation_predictions, all_param_scores, train_indices, holdout_indices.\n''')

    print('Return type is a dict. Check .keys() for details.')

    results_dict = {}
    results_dict['best_model'] = best_model
    results_dict['best_eval_folds'] = best_eval_folds
    results_dict['best_cv_fold'] = best_cv_fold
    results_dict['best_ntree_limit'] = best_ntree_limit
    results_dict['best_param'] = best_param
    results_dict['best_eval'] = best_eval
    results_dict['best_eval_stddev'] = best_eval_stddev
    results_dict['best_param_scores'] = best_param_scores
    results_dict['best_validation_predictions'] = best_validation_predictions
    results_dict['all_param_scores'] = all_param_scores
    results_dict['train_indices'] = train_indices
    results_dict['holdout_indices'] = holdout_indices
    if logfile is not None:
        sys.stdout=stdout_backup
    return results_dict
    
def getModelPoolStats(modelpool_dirs=['./model_pool'], metric=['auc','gini','binary_logloss','kaglloss']):
    curdir = os.getcwd()
    counter=0
    records = []
    columns = ['pool','model']
    first=True
    for folder in modelpool_dirs:
        for f in os.listdir(folder+'/history/'):
            hist = get(folder+'/history/'+ f)
            param = get(folder+'/param/'+f.split('param')[0]+'param'+f.split('param')[1].split('_')[0]+'.param')
            param = str(param)
            rec = []
            rec.append(counter)
            rec.append(f.split('.')[0])
            
            for m in metric:

                if first:
                    columns.append('val-'+m)
                    
                if m not in hist['val'].keys():
                    rec.append('-')
                else:
                    
                    if 'loss' in m:
                        rec.append(min(hist['val'][m]))
                    else:
                        rec.append(max(hist['val'][m]))
                
            for m in metric:

                if first:
                    columns.append('tr-'+m) 
                if m not in hist['train'].keys():
                    rec.append('-')
                else:
                    if 'loss' in m:
                        rec.append(min(hist['train'][m]))
                    else:
                        rec.append(max(hist['train'][m])) 
            if first:
                columns.append('param')
            rec.append(param)
            records.append(rec)
            first=False
        counter+=1
            
    df = pd.DataFrame(records)
    df.columns = columns
    
    return df          