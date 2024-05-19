import pandas as pd
import json
import os
from tqdm import tqdm
import numpy as np
from xml.etree import ElementTree as ET
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

current_file_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from __init__ import CriteriaModel
from __init__ import partition_criteria
from __init__ import wrapper_get_sentence_embedding
from __init__ import get_enrollment_difficulty
from __init__ import CriteriaModel



# trial_df = pd.read_csv(f'{current_file_path}/data/trial_data.csv', sep='\t')


# for row_idx, trial_row in tqdm(trial_df.iterrows(), total=len(trial_df)):

#     criteria = trial_row['criteria']
#     inclusion_criteria, exclusion_criteria = partition_criteria(criteria)
    
#     print(criteria)
    
#     print("inclusion: \n", inclusion_criteria)
#     print(len(inclusion_criteria))
#     print("exclusion: \n", exclusion_criteria)
#     print(len(exclusion_criteria))
    
#     sentence_num = len(inclusion_criteria) + len(exclusion_criteria)

def split_test():
    trial_df = pd.read_csv(f'{current_file_path}/data/trial_data.csv', sep='\t')
    trial_df['sentence_num'] = trial_df['criteria'].apply(lambda x: len(partition_criteria(x)[0]) + len(partition_criteria(x)[1]))

    # Split trial_df into two groups based on sentence_num
    group_gt_5 = trial_df[trial_df['sentence_num'] > 5]
    group_lte_5 = trial_df[trial_df['sentence_num'] <= 5]

    # Print the lengths of the two groups
    print("Number of rows with sentence_num > 5:", len(group_gt_5))
    print("Number of rows with sentence_num <= 5:", len(group_lte_5))


def plot_sentence_num(trial_df):
    
    import matplotlib.pyplot as plt


    trial_df['inclusion_num'] = trial_df['criteria'].apply(lambda x: len(partition_criteria(x)[0]))

    trial_df['exclusion_num'] = trial_df['criteria'].apply(lambda x: len(partition_criteria(x)[1]))

    plt.figure(figsize=(10, 6))
    plt.hist(trial_df['inclusion_num'], bins=range(0, max(trial_df['inclusion_num']) + 1, 1), alpha=0.7, color='b', label='Inclusion')
    plt.hist(trial_df['exclusion_num'], bins=range(0, max(trial_df['exclusion_num']) + 1, 1), alpha=0.7, color='r', label='Exclusion')
    plt.xlabel('Number of Sentences')
    plt.ylabel('Frequency')
    plt.title('Distribution of Inclusion and Exclusion Sentence Numbers')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('plots/plot.png')
    

    # Create a new column 'inclusion_num' in trial_df containing the number of sentences in 'inclusion_criteria'
    trial_df['inclusion_num'] = trial_df['criteria'].apply(lambda x: len(partition_criteria(x)[0]))

    # Create a new column 'exclusion_num' in trial_df containing the number of sentences in 'exclusion_criteria'
    trial_df['exclusion_num'] = trial_df['criteria'].apply(lambda x: len(partition_criteria(x)[1]))

    # Calculate mean and quartiles for inclusion_num
    inclusion_stats = trial_df['inclusion_num'].describe(percentiles=[0.25, 0.5, 0.75])

    # Calculate mean and quartiles for exclusion_num
    exclusion_stats = trial_df['exclusion_num'].describe(percentiles=[0.25, 0.5, 0.75])

    print("Size of data:", len(trial_df))
    

    # Print the statistics
    print("Inclusion Statistics:")
    print("Mean:", inclusion_stats['mean'])
    print("1st Quartile:", inclusion_stats['25%'])
    print("2nd Quartile (Median):", inclusion_stats['50%'])
    print("3rd Quartile:", inclusion_stats['75%'])
    print()

    print("Exclusion Statistics:")
    print("Mean:", exclusion_stats['mean'])
    print("1st Quartile:", exclusion_stats['25%'])
    print("2nd Quartile (Median):", exclusion_stats['50%'])
    print("3rd Quartile:", exclusion_stats['75%'])


def get_training_testing_data(data):
    
    trial_df = data
    
    trial_outcome_df = pd.read_csv(f'{current_file_path}/data/IQVIA_trial_outcomes.csv')
    iqvia_nctid_set = set(trial_outcome_df['studyid'])
    poor_set = set(trial_outcome_df[trial_outcome_df['trialOutcome'] == 'Terminated, Poor enrollment']['studyid'])
    
    get_sentence_embedding = wrapper_get_sentence_embedding()
    
    #trial_df = trial_df.iloc[:1000]  #################
    
    trial_emb_list = []
    
    
    for row_idx, trial_row in tqdm(trial_df.iterrows(), total=len(trial_df)):
        
        nctid = trial_row['nctid']
        criteria = trial_row['criteria']
        drugs = trial_row['drugs'].split(';')
        diseases = trial_row['diseases'].split(';')

        drugs_emb = torch.mean(torch.stack([get_sentence_embedding(drug) for drug in drugs]), dim=0)
        diseases_emb = torch.mean(torch.stack([get_sentence_embedding(disease) for disease in diseases]), dim=0)
        
        inclusion_criteria, exclusion_criteria = partition_criteria(criteria)

        inclusion_criteria_emb = get_sentence_embedding('\n'.join(inclusion_criteria))
        exclusion_criteria_emb = get_sentence_embedding('\n'.join(exclusion_criteria))

        trial_emb_list.append(torch.cat((inclusion_criteria_emb, exclusion_criteria_emb, drugs_emb, diseases_emb), dim=0))

    trial_emb = torch.stack(trial_emb_list)
    torch.save(trial_emb, f'{current_file_path}/data/trial_emb.pt')

    print(trial_emb.shape)

    X_data = trial_emb
    y_data = []
    for row_idx, trial_row in tqdm(trial_df.iterrows(), total=len(trial_df)):
        nctid = trial_row['nctid']

        if nctid in poor_set:
            y_data.append(1)
        else:
            y_data.append(0)

    y_data = torch.tensor(y_data)

    print(f"len(X_data): {len(X_data)}")
    print(f"len(y_data): {len(y_data)}")

    return X_data, y_data


def train_test_eval(X_train, X_test, y_train, y_test):
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train.cpu().numpy()), y=y_train.cpu().numpy())
    weight_for_positives = class_weights[1] 

    pos_weight = torch.tensor([weight_for_positives]).to(device)
    print(pos_weight)
        
    
    class CriteriaDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = CriteriaDataset(X_train, torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = CriteriaDataset(X_test, torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    model = CriteriaModel().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  ##remove weight and do another test
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)

    num_epochs = 50
    best_auc = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test.to(device))
            y_pred_test = y_pred_test.cpu().numpy().flatten()

            auc_test = roc_auc_score(y_test, y_pred_test)

            if auc_test > best_auc:
                best_auc = auc_test
                print(f"Epoch {epoch}\tBest AUC: {auc_test}, saving model...")

                torch.save(model.state_dict(), f'{current_file_path}/data/enrollment_model.pt')
    # Final evaluation
    model.load_state_dict(torch.load(f'{current_file_path}/data/enrollment_model.pt'))

    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test.to(device))
        y_pred_test = nn.Sigmoid()(y_pred_test).cpu().numpy().flatten()

        auc_test = roc_auc_score(y_test, y_pred_test)
        acc_test = ((y_pred_test > 0.5) == y_test.numpy()).mean()
        recall_test = ((y_pred_test > 0.5) & (y_test.numpy() == 1)).sum() / (y_test.numpy() == 1).sum()

        print(f"AUC: {auc_test}, Accuracy: {acc_test}, Recall: {recall_test}")
        
        return auc_test, acc_test, recall_test


def split_data(trial_df, cmd):
    
    idx = None
    if cmd == "inclusion":
        idx = 0
    else:
        idx = 1
        
    trial_df[f'{cmd}_num'] = trial_df['criteria'].apply(lambda x: len(partition_criteria(x)[idx]))
    trial_df['octile'] = pd.qcut(trial_df[f'{cmd}_num'], q=8, labels=False)

    octile_dfs = [trial_df[trial_df['octile'] == i] for i in range(8)]

    octile_train_dfs = [df.sample(frac=0.8, random_state=0) for df in octile_dfs]
    octile_test_dfs = [df.drop(octile_train.index) for octile_train, df in zip(octile_train_dfs, octile_dfs)]

    trial_df_train = pd.concat(octile_train_dfs)
    trial_df_test_list = octile_test_dfs

    print(f"Split by {cmd}_num")
    #print("Total size: ", len(trial_df))
    #print("Size of training set: ", len(trial_df_train))
    #print("Size of testing sets:", [len(df) for df in trial_df_test_list])

    return trial_df_train, trial_df_test_list


'''
cmd = "inclusion" or "exclusion"
octile = 0~7
'''
def run(cmd, octile):
    
    trial_df = pd.read_csv(f'{current_file_path}/data/trial_data.csv', sep='\t')
    trial_df_train, trial_df_test_list = split_data(trial_df, cmd)
    
    print("avg", cmd, trial_df_test_list[octile][f"{cmd}_num"].mean())
    
    X_train, y_train = get_training_testing_data(trial_df_train)
    X_test, y_test = get_training_testing_data(trial_df_test_list[octile])
    auc_test, acc_test, recall_test = train_test_eval(X_train, X_test, y_train, y_test)
    return cmd, octile, auc_test, acc_test, recall_test

def save_results(results, file_name):
    results_df = pd.DataFrame(results, columns=['Command', 'Octile', 'AUC', 'Accuracy', 'Recall'])
    results_df.to_csv(file_name, index=False)

def read_results(file_name):
    return pd.read_csv(file_name)

def octile():
    results = []
    for i in range(8):
        print("iter:", i)
        #run("inclusion", i)
        cmd, octile, auc_test, acc_test, recall_test = run("inclusion", i)
        results.append((cmd, octile, auc_test, acc_test, recall_test))
    
    save_results(results, 'plots/results_inclusion.csv')
    
    
    results = []
    for i in range(8):
        print("iter:", i)
        #run("exclusion", i)
        cmd, octile, auc_test, acc_test, recall_test = run("exclusion", i)
        results.append((cmd, octile, auc_test, acc_test, recall_test))
    
    save_results(results, 'plots/results_exclusion.csv')

if __name__ == "__main__":
    
    
    # plot_sentence_num(trial_df)
    
    # X_data, y_data, inclusion_num_list, exclusion_num_list = get_train_test_data(trial_df)
    

    import matplotlib.pyplot as plt

    octiles = range(8) 
    auc = [0.7332442748091604, 0.6747923681257014, 0.7279647435897436, 0.712618830576898,
        0.7276379368468368, 0.6981405294213175, 0.7093874665454729, 0.7294790038993764]
    accuracy = [0.7289002557544757, 0.7443064182194618, 0.7767653758542141, 0.7279534109816972,
                0.7625418060200669, 0.8471760797342193, 0.9208015267175572, 0.9154488517745303]
    recall = [0.552, 0.3333333333333333, 0.5128205128205128, 0.5904761904761905,
            0.5113636363636364, 0.23076923076923078, 0.0, 0.0]


    plt.figure(figsize=(10, 6))
    plt.plot(octiles, auc, label='AUC', marker='o')
    plt.plot(octiles, accuracy, label='Accuracy', marker='o')
    plt.plot(octiles, recall, label='Recall', marker='o')
    plt.xlabel('Octile')
    plt.ylabel('Values')
    plt.title('Exclusion Performance Metrics Across Octiles')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/exclusion.png', format='png', dpi=300)
    
    

    octiles = range(8)
    auc = [0.7386035455636024, 0.7085265941306514, 0.7211309523809525, 0.7357371571766671,
        0.6897433200084157, 0.7498765432098765, 0.7095126988984087, 0.6603873530872372]
    accuracy = [0.799290780141844, 0.6520522388059702, 0.9250535331905781, 0.7951977401129944,
                0.8217270194986073, 0.8385964912280702, 0.7324649298597194, 0.75]
    recall = [0.4507042253521127, 0.6086956521739131, 0.0, 0.43636363636363634,
            0.20618556701030927, 0.35555555555555557, 0.5581395348837209, 0.4095238095238095]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(octiles, auc, label='AUC', marker='o', color='blue')
    plt.plot(octiles, accuracy, label='Accuracy', marker='o', color='green')
    plt.plot(octiles, recall, label='Recall', marker='o', color='red')
    plt.xlabel('Octile')
    plt.ylabel('Values')
    plt.title('Inclusion Model Performance Metrics Across Octiles')
    plt.legend()
    plt.grid(True)

    plt.savefig('plots/inclusion.png', format='png', dpi=300)

    plt.show()


