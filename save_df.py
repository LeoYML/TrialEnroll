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
import json
import pickle

def load_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

current_file_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_data():
    
    trial_outcome_df = pd.read_csv(f'{current_file_path}/data/IQVIA_trial_outcomes.csv')

    iqvia_nctid_set = set(trial_outcome_df['studyid'])
    poor_set = set(trial_outcome_df[trial_outcome_df['trialOutcome'] == 'Terminated, Poor enrollment']['studyid'])


    with open(f'{current_file_path}/data/trials/all_xml.txt', 'r') as f:
        trials_file_list = [line.strip() for line in f]

    date_dict = load_dict('data/date_dict.pkl')
    age_dict = load_dict('data/age_dict.pkl')
    str_dict = load_dict('data/str_dict.pkl')
    
    trial_data_list = []
    for trial_path in tqdm(trials_file_list):
        nctid = trial_path.split('/')[-1].split('.')[0]

        if nctid not in iqvia_nctid_set:
            continue

        try:
            root_xml = ET.parse(f"{current_file_path}/data/{trial_path}").getroot()
            criteria = root_xml.find('eligibility').find('criteria').find('textblock').text 
            if len(criteria) == 0:
                continue

            interventions = [i for i in root_xml.findall('intervention')]
            drug_interventions = [i.find('intervention_name').text.lower().strip() for i in interventions if i.find('intervention_type').text=='Drug']
            if len(drug_interventions) == 0:
                continue
            drugs = ';'.join(drug_interventions)

            conditions = [i.text.lower().strip() for i in root_xml.findall('condition')]
            if len(conditions) == 0:
                continue
            diseases = ';'.join(conditions)

            if nctid in poor_set:
                label = 1
            else:
                label = 0
            
            
            if nctid not in date_dict:
                print(f"{nctid} not found in date_dict")
                duration = -1
                start_date = ''
                completion_date = ''
            else:
                duration = date_dict[nctid][2]
                start_date = date_dict[nctid][0]
                completion_date = date_dict[nctid][1]
                
            
                
            if nctid not in age_dict:
                print(f"{nctid} not found in age_dict")
                min_age, max_age = -1, -1
            else:
                min_age, max_age = age_dict[nctid][0], age_dict[nctid][1]
                
            if nctid not in age_dict:
                print(f"{nctid} not found in str_dict")
                min_age, max_age = -1, -1
            else:
                gender, phase, condition, intervention_type, intervention_name = str_dict[nctid]["gender"], str_dict[nctid]["phase"],\
                    str_dict[nctid]["condition"], str_dict[nctid]["intervention"][0], str_dict[nctid]["intervention"][1]
            
        
            trial_data_list.append((nctid, criteria, drugs, diseases, label, start_date, completion_date, duration, min_age, max_age, gender, phase))
            
            

        except AttributeError:
            print(f"Don't have criteria or drug or diseases for {trial_path}")
        except Exception as e:
            raise e

    trial_df = pd.DataFrame(trial_data_list, columns=['nctid', 'criteria', 'drugs', 'diseases', 'label', 'start_date', 'completion_date', 'duration', 'min_age', 'max_age', 'gender', 'phase'])
    trial_df.to_csv(f'{current_file_path}/data/trial_data.csv', index=False, sep='\t')
        


def partition_criteria(criteria):
    lines = [line.strip() for line in criteria.lower().split('\n') if line.strip()]

    inclusion_criteria, exclusion_criteria = [], []
    
    # Use a flag to indicate whether we are currently reading inclusion or exclusion criteria
    reading_inclusion = False
    reading_exclusion = False
    
    for line in lines:
        # Check if the line is an inclusion or exclusion header
        if 'inclusion criteria' in line:
            reading_inclusion = True
            reading_exclusion = False
            continue
        elif 'exclusion criteria' in line:
            reading_inclusion = False
            reading_exclusion = True
            continue
        
        if reading_inclusion:
            inclusion_criteria.append(line)
        elif reading_exclusion:
            exclusion_criteria.append(line)
    
    return inclusion_criteria, exclusion_criteria


def wrapper_get_sentence_embedding():
    model_name = "dmis-lab/biobert-base-cased-v1.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name).to(device)

    def get_sentence_embedding(sentence):
        # Encode the input string
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Send inputs to the same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get the output from BioBERT
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(**inputs)
        
        # Obtain the embeddings for the [CLS] token
        # The [CLS] token is used in BERT-like models to represent the entire sentence
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().to('cpu')
        
        return cls_embedding

    return get_sentence_embedding



def get_training_testing_data():
    
    trial_df = pd.read_csv(f'{current_file_path}/data/trial_data.csv', sep='\t')
    trial_df = trial_df[['nctid', 'criteria', 'drugs', 'diseases']]
    trial_df = trial_df.dropna()
    
    print(len(trial_df))
    
    
    trial_outcome_df = pd.read_csv(f'{current_file_path}/data/IQVIA_trial_outcomes.csv')
    iqvia_nctid_set = set(trial_outcome_df['studyid'])
    poor_set = set(trial_outcome_df[trial_outcome_df['trialOutcome'] == 'Terminated, Poor enrollment']['studyid'])
    
    get_sentence_embedding = wrapper_get_sentence_embedding()
    
    
    trial_emb_list = []
    
    test_idx_list = []
    
    
    for row_idx, trial_row in tqdm(trial_df.iterrows(), total=len(trial_df)):
        
        if trial_row['nctid'] in test_id:
            print(row_idx)
            test_idx_list.append(row_idx)
        
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
    

    print(trial_emb.shape)
    
    torch.save(trial_emb, f'{current_file_path}/data/trial_emb_.pt')
    
    with open('data/test_ids.json', 'w') as f:
        json.dump(test_idx_list, f)


def load_data():
    
    trial_df = pd.read_csv(f'{current_file_path}/data/trial_data.csv', sep='\t')
    
    trial_outcome_df = pd.read_csv(f'{current_file_path}/data/IQVIA_trial_outcomes.csv')
    iqvia_nctid_set = set(trial_outcome_df['studyid'])
    poor_set = set(trial_outcome_df[trial_outcome_df['trialOutcome'] == 'Terminated, Poor enrollment']['studyid'])
    
    x_data = torch.load(f'{current_file_path}/data/trial_emb.pt')
    y_data = []
    for row_idx, trial_row in tqdm(trial_df.iterrows(), total=len(trial_df)):
        nctid = trial_row['nctid']

        if nctid in poor_set:
            y_data.append(1)
        else:
            y_data.append(0)

    y_data = torch.tensor(y_data)

    print(f"len(X_data): {len(x_data)}")
    print(f"len(y_data): {len(y_data)}")
    
    return x_data, y_data


test_id = [
    "NCT00088075", "NCT01584830", "NCT01695343", "NCT00541242", "NCT00735709",
    "NCT00514371", "NCT00459953", "NCT00614653", "NCT01652794", "NCT01148511",
    "NCT01369888", "NCT02195700", "NCT00153036", "NCT02835924", "NCT02273973",
    "NCT00334282", "NCT00219271", "NCT01003379", "NCT01617434", "NCT02544438",
    "NCT01529112", "NCT02688933", "NCT00631618", "NCT00347048", "NCT02083536",
    "NCT01098539", "NCT02032901", "NCT02176369", "NCT02309281", "NCT02552667",
    "NCT01833494", "NCT00124072", "NCT00288054", "NCT02003924", "NCT00130039",
    "NCT00664378", "NCT00329472", "NCT02468661", "NCT00063401", "NCT01158144"
]


if __name__ == "__main__":

    preprocess_data()
    
    #get_training_testing_data()

