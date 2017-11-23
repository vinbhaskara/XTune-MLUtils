'''
Created on Nov 18, 2017

@author: Vineeth_Bhaskara
'''
from __future__ import print_function
import numpy as np
import pandas as pd
import pickle
import os
import sys
import bcolz
import cv2
import random
from xtune import *
from sklearn.metrics import roc_auc_score, log_loss

def normalizedf(df):
    '''
    Normalizes dataframe columns to 1. Send in slice of those columns
    that have to be normalized.
    '''
    df = df.div(df.sum(axis=1), axis=0)
    return df
    
# Dealing with categorical features Way 1
def target_encode(trn_series=None, tst_series=None, target=None, in_samples_leaf=1,
                  smoothing=1, noise_level=0):
    """
    Target Encoding for Categorical Features! An alternative to OneHot Encoding. But with a risk of GT leakage.
    This converts Categorical to Numerical fields.
    
    Usage:
    Example - Do Column by Column

    for f in catcols:
        train_df[f + "_new"], test_df[f + "_new"] = target_encode(trn_series=train_df[f],
                                            tst_series=test_df[f],
                                            target=train_df['target'],
                                            min_samples_leaf=200,
                                            smoothing=10,
                                            noise_level=0) 
    
    min_samples_leaf define a threshold where prior and target mean (for a given category value) have the same weight. 
    Below the threshold prior becomes more important and above mean becomes more important.

    How weight behaves against value counts is controlled by smoothing parameter
    
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    def add_noise(series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))
    
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
    
    
# Dealing with Categorical features Way 2    
def doOneHot(df1, ranges):
    '''
    Ready made column one hot function. Use for categorical features especially when using tree based classifiers.
    Pass in the ranges in the format list of list [[column_name, range_list],...]
    Pass in the df.    
    '''
    df = df1.copy() 
    catcolumns = [i[0] for i in ranges]
    print('Original shape: ', df.shape)
    dummydf = df.head(1).copy()   
    dummydf_full = None
    
    for i in ranges:
        for j in i[1]:
            dummydf[i[0]] = j
            if dummydf_full is None:
                dummydf_full = dummydf.copy()
            else:
                dummydf_full = dummydf_full.append(dummydf.copy())

    
    #dummydf_full = pd.DataFrame(dummyrows)
    #dummydf_full.columns = df.columns
    print('Dummy rows shape: ', dummydf_full.shape)
    print('One Hotting...')
    
    df['type'] = 'data'
    dummydf_full['type'] = 'dummy'
    dfappended = df.append(dummydf_full)
    
    dfonehot = pd.get_dummies(dfappended, columns=catcolumns)
    
    finaldf = dfonehot[(dfonehot['type']=='data')]
    
    del finaldf['type']
    print('One hot done. New shape: ', finaldf.shape)
    
    return finaldf.copy() 
    
    
def RankAverager(valpreds, testpreds, predcol='pred', scale_test_proba=False):
    '''
    Expects <predcol> as the column for prediction values that need be ranked ascending wise - say Class 1. 
    Suitable for Binary class problems.
    Output ranks are displayed in column 'rank_avg_<predcol>' and 'rank_avg_<predcol>_proba' for Class 1.
    
    scale_test_proba scales the rank averaged probabilities from 0 to 1 so the minimum is set 0 and the maximum
    goes to 1 in the Test set. So it is very important to only use this iff your tests predictions have
    a lot of rows. This is generally used in case you are going to average the output probs with some
    other model later.
    
    Very efficient implementation! Uses Binary Search.
    '''
    print('Val: ', valpreds.shape)
    print('Test: ',testpreds.shape)
    
    valpreds['rank_type'] = 'validation'
    testpreds['rank_type'] = 'test'
    
    valpreds = valpreds.sort_values(by=predcol, ascending=True)
    
    valpreds.reset_index(drop=True, inplace=True)
    valpreds['rank_id'] = np.arange(valpreds.shape[0])
    
    testpreds.reset_index(drop=True, inplace=True)
    testpreds['rank_id'] = np.arange(testpreds.shape[0])
    
    rank_avgs = valpreds[predcol].searchsorted(testpreds[predcol]) + 1
    
    testpreds['rankavg_'+predcol]  = rank_avgs  
    
    if scale_test_proba:
        proba = np.array(rank_avgs)/float(valpreds.shape[0]+1)        
        proba = proba - np.min(proba)
        proba = proba/np.max(proba)        
        testpreds['rankavg_'+predcol+'_proba']  = proba
    else:
        testpreds['rankavg_'+predcol+'_proba']  = np.array(rank_avgs)/float(valpreds.shape[0]+1)
    
    return testpreds.copy() 
    
def preds_averager(preds, weights=None, type='AM'):
    '''
    preds is a list of predictions from predictors(numpy) taken directly from 
    predict methods of classifiers.
    
    AM - Arithmetic Mean
    GM - Geometric Mean
    HM - Harmonic Mean (weights not supported yet)
    
    No renormalization of preds is done.
    '''
    
    if weights is None:
        weights = np.ones((5,), dtype=np.int)
    
    if len(preds) == 1:
        return preds[0]
    elif len(preds) == 0:
        return None
    
    ans = []
    if type=='AM':
        ans = preds[0]*weights[0]
        counter = 1
        for i in preds[1:]:
            ans = ans + i*weights[counter]
            counter += 1
        
        ans = ans/float(sum(weights))
        
    if type=='GM':
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        ans = preds[0]**weights[0]
        counter = 1
        for i in preds[1:]:
            ans = ans*(i**weights[counter])
            counter +=1
        
        
    if type== 'HM': # no weights supported currently
        
        invans = 1.0/preds[0]
        counter = 1
        for i in preds[1:]:
            invans = invans + 1.0/i
            counter+=1
        ans = float(len(preds))/invans
    
    
    return ans
    
    
def desperateFitter(dflist, predcols=['pred'], gtcol='target', thrustMode=False, niters=1000, 
                    metric=['logloss','gini','auc'], is_more_better=True, coarseness=10, custom_weight_functions=[np.exp]):
    '''
    DesperateFitter v1.12 - If you are desperate enough to not try a regression model!
    Iterates through Random weights that sum up to 1 and maximize a 
    measure on a target.
    
    dflist: is the validation predictions of various models (type list)
    
    metric: you may pass multiple metrics as functions in a list. But the best pick would be based on the 
    last member of the list for which is_more_better is applicable
    
    custom_weight_functions: use this to pass a function that weights the individual model scores
    supported only for auc and logloss
    
    thrustMode: When true, takes models pairwise and calculates weights successively greedily starting by
    fitting the best models and then the lesser good models. 
    
    '''
    from sklearn.metrics import roc_auc_score, log_loss
    if metric[-1] == 'auc':
        is_more_better = True
    elif metric[-1] == 'logloss':
        is_more_better = False
        
    def calcMetrics(dfs, weight, silent=True):
        newdf = dfs[0].copy()
        
        for predcol in predcols:
            newdf[predcol] = dfs[0][predcol]*weight[0]
            for j in range(1, len(dfs)):
                newdf[predcol] += dfs[j][predcol]*weight[j]
            
        metric_results = []
        metric_labels = []
        
        if not silent:
            print(weight ,end='  ')
        counter = 0
        for i in metric:
            metric_label = None
            if i == 'auc':
                metric_label='auc'
                i=roc_auc_score
            elif i=='logloss':
                metric_label='ll'
                i=log_loss
            elif i=='gini':
                metric_label='gini'
                i=eval_gini
            else:
                metric_label='metric'+str(counter)
                
            if len(predcols)==1:
                metric_result = i(newdf['target'], newdf[predcols[0]])   
            else:
                metric_result = i(newdf['target'], newdf[predcols])    
            metric_results.append(metric_result)
            metric_labels.append(metric_label)
            
            if not silent:
                print(metric_label, ':', metric_result, '\t', end='')
            
            counter+=1
        
        return metric_labels, metric_results
    
    
    nummodels = len(dflist)   
    
    if nummodels >= coarseness:
        print('Coarness set is not compatible to the number of models input. Adjusting...')
        coarseness = nummodels*3
        
    
    init_weights = np.eye(nummodels).tolist()
    init_weights.append((np.ones((1, nummodels))/float(nummodels)).tolist()[0])
    
   
    print('Init Metrics: ')
    init_metrics = []    
    all_init_metrics = []
    for wt in init_weights:
  
        label, metric_results = calcMetrics(dflist, wt, silent=False)                       
        init_metrics.append(metric_results[-1]) # evaluation based on only the last metric
        all_init_metrics.append(metric_results)
        print('\n')
        
    best_metric = None
    wt_index=None
    if is_more_better:
        best_metric = max(init_metrics)
    else:
        best_metric = min(init_metrics)
    best_weights = init_weights[init_metrics.index(best_metric)]
    
    
    print('\n\nEvaling custom metrics')
    # add custom functions to check performances when weighted by indiv model score
    custom_weights = []
    custom_metrics = []
    
    # identity function
    identityfunc = lambda x: x
    inversefunc = lambda x: 1.0/np.abs(x)
    
    if 'logloss' in metric:
        loglosses = []
        # collect loglosses
        
        for i in range(nummodels):
            loglosses.append(all_init_metrics[i][metric.index('logloss')])
        loglosses = np.array(loglosses)
        
        for wtfunction in custom_weight_functions + [inversefunc]:
            wts = wtfunction(-1.0 * loglosses)
            
            custom_weights.append(wts/np.sum(wts))
    
    if 'auc' in metric:
        aucs = []
        # collect aucs
        
        for i in range(nummodels):
            aucs.append(all_init_metrics[i][metric.index('auc')])
        aucs = np.array(aucs)
        
        for wtfunction in custom_weight_functions + [identityfunc]:
            wts = wtfunction(aucs)
            custom_weights.append(wts/np.sum(wts))
            
    
    for wt in custom_weights:
  
        label, metric_results = calcMetrics(dflist, wt, silent=False)                       
        custom_metrics.append(metric_results[-1]) # evaluation based on only the last metric
        print('\n')
        
    custom_best_metric = None
    if is_more_better:
        custom_best_metric = max(custom_metrics)
        if custom_best_metric>best_metric:
            best_metric = custom_best_metric
            best_weights = custom_weights[custom_metrics.index(custom_best_metric)]
    else:
        custom_best_metric = min(custom_metrics)
        if custom_best_metric<best_metric:
            best_metric = custom_best_metric
            best_weights = custom_weights[custom_metrics.index(custom_best_metric)] 
    
    
    
    print('The best eval metric on initial weights is: ', best_metric, ' with weights: ', 
          best_weights, '\n')            
       
    print('\nStarting random search for weights... \n')
    
    if not thrustMode:
        
        for iters in range(niters):
            randnums = []
            for x in range(nummodels):
                randnums.append(random.randint(1, int(coarseness)))
            randnums = np.array(randnums)

            randweights = randnums/np.sum(randnums)
            metric_labels, metric_results = calcMetrics(dflist, randweights, silent=True)
            
            if is_more_better:
                if metric_results[-1] > best_metric:
                    best_metric = metric_results[-1]
                    best_weights = randweights
                    
                    print(randweights, end='  ')
                    for i in range(len(metric_labels)):
                        print(metric_labels[i], ':', metric_results[i], '\t', end='')
                    print('\n')
            else:
                if metric_results[-1] < best_metric:
                    best_metric = metric_results[-1]
                    best_weights = randweights
                    
                    print(randweights, end='  ')
                    for i in range(len(metric_labels)):
                        print(metric_labels[i], ':', metric_results[i], '\t', end='')
                    print('\n')
                        
    else: # thrust mode!
        
        # First sort the models based on their individual goodness
        model_priority = []
        init_metrics_top = init_metrics[:-1].copy()
        
        init_metrics_top_sorted = init_metrics_top.copy()
        if is_more_better:
            init_metrics_top_sorted.sort(reverse=True)
        else:
            init_metrics_top_sorted.sort()
            
        for i in init_metrics_top_sorted:
            model_priority.append(init_metrics_top.index(i))
        
        print('\nModel Priority: ', model_priority)
        
        
        df1 = dflist[model_priority[0]].copy()
        print('Thrusting round 1: Models: ', model_priority[0], ' and ', model_priority[1], '\n')
        
        best_thrust_weights = []
        for i in range(1, nummodels):
            if i!=1:
                print('\nIncremental Thrusting ',i,' with Model', model_priority[i],'\n')
            df2 = dflist[model_priority[i]].copy()
            
            goodweights12 = None 
            goodeval = None
            for weight1 in np.arange(0, 1, 1.0/coarseness)[::-1]:
                
                weights12 = [weight1, 1.0-weight1]
                metric_labels, metric_results = calcMetrics([df1, df2], weights12, silent=True)

                if goodeval is None:
                    goodeval = metric_results[-1]
                    goodweights12 = weights12
                    print(goodweights12, end='  ')
                    for i in range(len(metric_labels)):
                        print(metric_labels[i], ':', metric_results[i], '\t', end='')
                    print('\n')
                    
                else:
                    
                    if is_more_better:
                        if metric_results[-1] > goodeval:
                            goodeval = metric_results[-1]
                            goodweights12 = weights12

                            print(goodweights12, end='  ')
                            for i in range(len(metric_labels)):
                                print(metric_labels[i], ':', metric_results[i], '\t', end='')
                            print('\n')
                    else:
                        if metric_results[-1] < goodeval:
                            goodeval = metric_results[-1]
                            goodweights12 = weights12

                            print(goodweights12, end='  ')
                            for i in range(len(metric_labels)):
                                print(metric_labels[i], ':', metric_results[i], '\t', end='')
                            print('\n')
                            
            best_thrust_weights.append(goodweights12)
            
            df1[predcols] = df1[predcols]*goodweights12[0] + df2[predcols]*goodweights12[1]
            
        
        print(best_thrust_weights)
        
        final_thrust_weights = np.array(best_thrust_weights[0].copy())
        
        for j in range(1, len(best_thrust_weights)):
            
            final_thrust_weights = final_thrust_weights*best_thrust_weights[j][0]
            final_thrust_weights = final_thrust_weights.tolist()
            final_thrust_weights.append(best_thrust_weights[j][1])
            final_thrust_weights = np.array(final_thrust_weights)
            
        final_thrust_weights = final_thrust_weights.tolist()
        
        # get the final thrust weights in the right order 
        final_thrust_weights_right_order = []
        
        for i in model_priority:
            final_thrust_weights_right_order.append(final_thrust_weights[i])        
        
        best_weights = final_thrust_weights_right_order
        
    
    metric_labels, metric_results = calcMetrics(dflist, best_weights, silent=True)

    print('\n\nThe best eval is: ', metric_labels, ' - ', metric_results, ' with weights: ', 
      best_weights, '\n')   
    return best_weights, metric_labels, metric_results


def gaussian_feature_importances(df, missing_value=-1, skip_columns=['id','target']):
    '''
    If missing_value is provided then fields having -1 are neglected while generating feature importances.
    Assuming binary target. 
    df should have target column.
    skip columns from features can be provided.
    
    Remember Stat Mech course of Prof Nandy?
    '''
    
    cols = df.columns
    results = []
    
    for i in cols:
        if i in skip_columns:
            continue
        
        std0 = df[(df[i]!=missing_value) & (df['target']==0)][i].std()
        mean0 = df[(df[i]!=missing_value) & (df['target']==0)][i].mean()
        std1 = df[(df[i]!=missing_value) & (df['target']==1)][i].std()
        mean1 = df[(df[i]!=missing_value) & (df['target']==1)][i].mean()
        diff_mean = abs(mean1-mean0)
        diff_std = abs(std1-std0)
        try:
            significance_measure = diff_mean/diff_std
        except:
            significance_measure = None
        
        results.append([i, mean0, std0, mean1, std1, diff_mean, diff_std, significance_measure])
        
    results = pd.DataFrame(results)
    results.columns = 'col,mean0,std0,mean1,std1,diff_mean,diff_std,significance_measure'.split(',')
    
    return results.sort_values(by=['significance_measure', 'diff_mean'], ascending=False, kind='mergesort', na_position='first')
    
    
''' create feature interactions pairwise on train, and use those decided newfeature pickups for test using custom_ops'''
def create_pairwise_feature_interactions(df, custom_ops=[], columns=None, type='multiplicative', skip_cols=[]):
    ''' 
    Checks the statistical significance of new multiplicative feature combinations from the columns set given.
    If no columns are specified, then all combinations of the columns excluding the id and target fields are considered.
    id fields is considered to be some non feature column
    target fields is considered to be the GT column
    Specify more skip columns by skip_columns
    type: 'multiplicative' or 'additive' or 'both'
    
    custom_ops: you can specify a list of strings that ask for certain particular custom interaction features to
    be added. Example: newfeature-F1-F2_add, newfeature_F1-F2_mul will create F1+F2 feature, F1*F2 respectively.
    
    '''
    
    newdf = pd.DataFrame()
    
    if 'target' in df.columns:
        newdf['target'] = df['target']
    
    if len(custom_ops)!=0:
        for i in custom_ops:
            
            if i.split('|')[0] == 'newfeature':
                operation = i.split('|')[-1]
                f1 = i.split('|')[1]
                f2 = i.split('|')[2]

                if operation == 'add':
                    newdf[i] = df[f1] + df[f2]
                elif operation == 'mul':
                    newdf[i] = df[f1] * df[f2]
            else:
                newdf[i] = df[i]
        
        if 'target' in newdf.columns:
            return newdf, feature_importances(newdf)    
        else: 
            return newdf
            
    
    if columns is None:
        givencolumns = df.columns
        columns = []
        for i in givencolumns:
            if i!='id' and i!='target' and i not in skip_cols:
                columns.append(i)
    
    # keep the original columns intact
    for i in columns:
        newdf[i] = df[i]
    
    counter1 = 0
    for i in range(len(columns)):
        percent = (counter1*100)/float(len(columns))
        if int(percent) % 2 == 0 and int(percent)!=0:
            print('Progress: ', percent, ' %', end='')
        
        if i == len(columns) -1 :
            break
        for j in range(i+1, len(columns)):
            
            if type=='multiplicative' or type=='both':
                newdf['newfeature|'+columns[i]+'|'+columns[j]+'|mul'] = df[columns[i]] * df[columns[j]]
                
            if type=='additive' or type=='both':
                newdf['newfeature|'+columns[i]+'|'+columns[j]+'|add'] = df[columns[i]] + df[columns[j]]
                
        counter1+=1
        
        
    if 'target' in newdf.columns:
        return newdf, feature_importances(newdf)    
    else:
        return newdf

# easy pickles
def get(name):
    f=open(name, 'rb')
    try:
        mod = pickle.load(f)
    except:
        mod = pickle.load(f, encoding='latin1') # to resolve python 2 vs 3 incompatibility of pickles
    f.close()
    return mod

def put(path, obj):
   f=open(path, 'wb')
   pickle.dump(obj, f)
   f.close()
   return True

# saving huge arrays without loss of precision or disk size constraint
def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]



# Criag Glastonbury 23rd on PB LB (0.5 logloss) did a preprocessing of normalizing the RGB histogram with
# Obvly it gave him great results than us (~1 logloss) :/
def normalized(rgb):
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)
    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]
    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)
    return norm

def get_im_cv2(path):
    img = cv2.imread(path,1)
    
    # For color historgram
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # Sharpen
    output_3 = cv2.filter2D(img_output, -1, kernel_sharpen_3)
    
    # Reduce to manageable size
    resized = cv2.resize(output_3, (224, 224), interpolation = cv2.INTER_LINEAR)
    return resized
    