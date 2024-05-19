import pandas as pd
import json
import os
from tqdm import tqdm
import numpy as np
import lightgbm as lgb
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

current_file_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from __init__ import partition_criteria

trial_df = pd.read_csv(f'{current_file_path}/data/trial_data.csv', sep='\t')

trial_df['inclusion_num'] = trial_df['criteria'].apply(lambda x: len(partition_criteria(x)[0]))
trial_df['exclusion_num'] = trial_df['criteria'].apply(lambda x: len(partition_criteria(x)[1]))
trial_df['sentence_num'] = trial_df['criteria'].apply(lambda x: len(partition_criteria(x)[0]) + len(partition_criteria(x)[1]))

#print(len(trial_df))


def train(training_set):

    X = training_set
    y = trial_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    gbm = lgb.train(
        params,
        dtrain,
        num_boost_round=100,
        valid_sets=[dtrain, dtest],
        valid_names=['train', 'test'],
        callbacks=[lgb.early_stopping(10)]
    )

    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    y_pred_binary = (y_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    #print(f'Accuracy of the model: {accuracy}')

    auc_roc = roc_auc_score(y_test, y_pred)
    #print(f"AUC-ROC Score: {auc_roc}")

    return auc_roc, accuracy


if __name__ == "__main__":
    
    features = [
        trial_df[['exclusion_num']],
        trial_df[['inclusion_num']],
        trial_df[['inclusion_num', 'exclusion_num']],
        trial_df[['sentence_num', 'inclusion_num', 'exclusion_num']],
    ]


    result = [
        ['exclusion_num'],
        ['inclusion_num'],
        ['inclusion_num + exclusion_num'],
        ['sentence_num + inclusion_num + exclusion_num']
    ]
    
    result = {
        'feature': ['exclusion_num', 'inclusion_num', 'inclusion_num + exclusion_num', 'sentence_num + inclusion_num + exclusion_num'],
        'auc_roc': [None, None, None, None],
        'accuracy': [None, None, None, None]       
    }

    num_iteration = 5
    
    for i in range(len(features)):
        
        total_auc_roc, total_accuracy = 0, 0
        
        for j in range (num_iteration):
            auc_roc, accuracy = train(features[i])
            total_auc_roc += auc_roc
            total_accuracy += accuracy
        
        avg_auc_roc, avg_accuracy = total_auc_roc / num_iteration, total_accuracy / num_iteration
        
        result["auc_roc"][i] = avg_auc_roc
        result["accuracy"][i] = avg_accuracy
        
    print(result)
    
    df = pd.DataFrame(result)

    # Save DataFrame to a CSV file
    df.to_csv('results/inclusion_exclusion_features.csv', index=False)
    
    
    
        
        