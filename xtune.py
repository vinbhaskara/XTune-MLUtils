from __future__ import print_function

import os
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss
import sys, gc
import matplotlib.pyplot as plt 
'''
TODOS -
Caliberating the predictions
Gaussian optimization of the parameters 
MP support across param grid
'''

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


def kag_auc(pred, d_eval):
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


    
def xPredict( model, d_pred):
    return model.predict(d_pred, ntree_limit=model.best_ntree_limit)

def gcRefresh():
    gc.collect()


def xTrain( d_train, param, val_data=None, prev_model=None, verbose_eval=True):
    '''
    Usage:
        1) d_train: xgb DMatrix of Train data
        2) param: Training parameters
               Example default param -
                param_xgb_default={
                'booster':'gbtree',
                'silent':0, 
                'num_estimators':1000,
                'early_stopping':5,
                'eval_metric':'mlogloss', # if feval is set then that overrides eval_metric
                'objective':'multi:softprob',
                'num_class':2,
                'feval':'kag_auc', # feval overrides eval_metric for early stopping. You may pass custom functions too.
                'maximize_feval': True
                }
        3) val_data: xgb DMatrix for validation data 
        4) prev_model: for continuing training
        5) verbose_eval: True/False - To display individual boosting rounds stats
        
        Returns: trained model, and history dictionary
    '''

    if param is None:
        print('No param passed. Check an example: ', xtrain.__doc__)
        sys.exit()

    param_xgb = param.copy() 

    if param_xgb['feval'] == 'kag_auc':
        param_xgb['feval'] = kag_auc

    if 'num_estimators' not in list(param_xgb.keys()):
        print('Choosing default num_estimators: ', 5000)
        param_xgb['num_estimators'] = 5000
    if 'early_stopping' not in list(param_xgb.keys()):
        param_xgb['early_stopping'] = 5
    if 'feval' not in list(param_xgb.keys()):
        param_xgb['feval'] = None


    if val_data is None:
        watchlist=[(d_train, 'train')]
        if param_xgb['early_stopping'] is not None:
            print('Ignoring early stopping as no validation data passed.')
            param_xgb['early_stopping']=None
    else:
        watchlist=[(d_train, 'train'),(val_data, 'val')]

    history_dict={}
    feval=None
    if param_xgb['feval'] is not None:
        feval=param_xgb['feval']
    else:
        param['maximize_feval'] = False
    xgb_model=xgb.train(param_xgb, d_train, num_boost_round=param_xgb['num_estimators'], evals=watchlist,
             feval=feval, maximize=param_xgb['maximize_feval'], # custom metric for early stopping
             early_stopping_rounds=param_xgb['early_stopping'], 
             evals_result=history_dict,
             xgb_model=prev_model, # allows continuation of previously trained model
             verbose_eval=verbose_eval)
    return xgb_model, history_dict.copy()



def xGridSearch( d_train, params, randomized=False, num_iter=None, rand_state=None, isCV=True, 
              folds=5, d_holdout=None, verbose_eval=True):
    '''       

    Usage:
        1) d_train: xgb DMatrix Training Data
        2) params: dictionary with list of possible values for a parameter as keys
                   Example params:
                    params_xgb_default_dict={
                        'booster':['gbtree'],
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
                        'num_class':[2],
                        'feval':['kag_auc'], # feval overrides eval_metric for early stopping. You may pass custom functions too.
                        'maximize_feval': [True]
                        }
        3) randomized: False/True - To randomly choose points from the parameter Grid for Search (without replacement)
        4) num_iter: Specified when randomized=True to limit the total number of random parameters considered. If None,
        run continues till the parameter grid is exhausted
        5) rand_state: for reproducibility
        6) isCV: True/False - Cross Validation OR Holdout
        7) folds: If isCV, then the number of CV folds, else represents the holdout portion to be 1/folds fraction of the
        training data provided given d_holdout is None
        8) d_holdout: Data for holdout. If specified, then folds has no effect.
        9) verbose_eval: True/False - verbosity - printing each round stats

    Note:
        If isCV is True does Cross Validation (Stratified) for folds times over d_train data.
        If isCV is False, then does a holdout by taking the d_holdout data.
        If isCV is False and d_holdout is None, then one stratified split of (100-100/folds):100/folds 
        is made train/holdout, and returns the holdout indices.


    Return Value: A dictionary with the following keys:
        best_param, best_eval, best_eval_folds, all_param_scores, *best_model, best_ntree_limit,
        holdout indices, train indices, best_validation_predictions, best_plots_lst

    *best_model would be the a list of models across folds of the best parameter.

    '''

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

    if not isCV and d_holdout is None:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rand_state)
        print('Making a ', 100-100//folds, ' and ', 100//folds, ' split of Train:Test for Holdout.')
        for tr, ts in skf.split(np.zeros(len(d_train.get_label().tolist())), d_train.get_label()):
            d_holdout = d_train.slice(ts)
            d_train = d_train.slice(tr)
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
        allparams = pg[indices]
    else:
        allparams = pg

    total = len(allparams)
    counter=0
    for param in allparams:
        counter+=1

        best_ntree_limit_folds=[]
        best_ntree_score_folds=[]
        ntree_hist_scores_folds=[]
        best_model_lst=[]

        val_pred=None # validation predictions of this param

        print('\n')
        print('#######################################################################')

        is_eval_more_better = False # error metric assumed default
        if param['feval'] is not None and param['maximize_feval'] is None:
            print('If want to use feval you must set maximize_feval. Now continuing without feval.')
        elif param['feval'] is not None and param['maximize_feval'] == True:
            is_eval_more_better = True
            print('The eval metric is being maximized.')

        if param['feval'] is None and param['eval_metric']=='auc':
            is_eval_more_better = True
            print('The eval metric is being maximized.')

        if not is_eval_more_better:
            print('The eval metric is being minimized.')


        print('Doing param ', counter, ' of total ', total,' - ', param, '\n')
        if not isCV: # holdout set 

            model, hist = xTrain(d_train, param, d_holdout, verbose_eval=verbose_eval)
            val_pred = model.predict(d_holdout, ntree_limit=model.best_ntree_limit)

            print('Holdout: Score ', model.best_score, ' Trees ',model.best_ntree_limit)
            print('\n')

            best_ntree_limit_folds.append(model.best_ntree_limit)
            best_ntree_score_folds.append(model.best_score)
            ntree_hist_scores_folds.append(hist)              
            best_model_lst.append(model)


        else: # cross-validation     

            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=rand_state)
            skf_split = skf.split(np.zeros(len(d_train.get_label().tolist())), d_train.get_label())
            foldcounter=0
            for tr, ts in skf_split:
                foldcounter+=1
                print('Doing CV fold #', foldcounter)
                xgb_train_cv = d_train.slice(tr)
                xgb_val_cv = d_train.slice(ts)

                model, hist = xTrain(xgb_train_cv, param, xgb_val_cv, verbose_eval=verbose_eval)
                val_pred_fold = model.predict(xgb_val_cv, ntree_limit=model.best_ntree_limit)

                if val_pred is None:
                    val_pred=np.zeros((int(len(d_train.get_label().tolist())), val_pred_fold.shape[1]))
                val_pred[ts]=val_pred_fold

                print('CV Fold: Score ', model.best_score, ' Trees ',model.best_ntree_limit)
                print('\n')

                best_ntree_limit_folds.append(model.best_ntree_limit)
                best_ntree_score_folds.append(model.best_score)
                ntree_hist_scores_folds.append(hist)
                best_model_lst.append(model)


        if is_eval_more_better:
            best_score_across_folds=max(best_ntree_score_folds)
            best_fold = best_ntree_score_folds.index(max(best_ntree_score_folds))
        else:
            best_score_across_folds=min(best_ntree_score_folds)
            best_fold = best_ntree_score_folds.index(min(best_ntree_score_folds))
        best_ntree_limit_across_folds=best_ntree_limit_folds[best_fold]

        all_param_scores.append([param.copy(), {'ntree_limit_folds': best_ntree_limit_folds, \
                                                'best_ntree_score_folds': best_ntree_score_folds, \
                                                'ntree_hist_scores_folds': ntree_hist_scores_folds, \
                                                'score_avg': sum(best_ntree_score_folds)/float(len(best_ntree_score_folds)), \
                                                'best_ntree_limit_across_folds': best_ntree_limit_across_folds, \
                                                'best_score_across_folds': best_score_across_folds, \
                                                'best_fold': best_fold}])

        update=False
        if best_eval is None:
            update=True
        else:
            if is_eval_more_better:
                if model.best_score > best_eval:
                    update=True
            else:
                if model.best_score < best_eval:
                    update=True

        if update:
            best_eval = sum(best_ntree_score_folds)/float(len(best_ntree_score_folds))
            best_param = param.copy()
            best_ntree_limit = best_ntree_limit_across_folds
            best_validation_predictions=val_pred
            best_model = best_model_lst
            best_param_scores=all_param_scores[-1]
            best_cv_fold=best_fold
            best_eval_folds=best_ntree_score_folds
            best_iteration=best_ntree_limit-1 # iteration is counted from 0 whereas trees from 1 (obvly)

        print('Params: ',param, '\nCV Scores: ', best_ntree_score_folds, ' \nAvg CV Score: ', sum(best_ntree_score_folds)/float(len(best_ntree_score_folds)), \
        '\nBest Fold: ', best_fold, '\nNumTreesForBestFold: ', best_ntree_limit_across_folds)

    print('\n')
    print('***********************************************************************')
    print('Final Results\n')
    print('Best Params: ',best_param, '\nCV Scores: ', best_eval_folds, ' \nAvg CV Score: ', best_eval, '\nBest Fold: ', \
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
    results_dict['best_param_scores'] = best_param_scores
    results_dict['best_validation_predictions'] = best_validation_predictions
    results_dict['all_param_scores'] = all_param_scores
    results_dict['train_indices'] = train_indices
    results_dict['holdout_indices'] = holdout_indices

    return results_dict


