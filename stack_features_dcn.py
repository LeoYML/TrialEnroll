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

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

current_file_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


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


def pad_sentences(paragraphs, padding_size):
    padded_paragraphs = []
    mask_matrices = []

    for p in paragraphs:
        num_padding = padding_size - p.size(0)
        
        if num_padding > 0:
            padding = torch.zeros(num_padding, p.size(1))
            padded_p = torch.cat([p, padding], dim=0)
        else:
            padded_p = p

        # 1 for actual data, 0 for padding
        mask = torch.cat([torch.ones(p.size(0)), torch.zeros(num_padding)], dim=0)

        padded_paragraphs.append(padded_p)
        mask_matrices.append(mask)

    padded_paragraphs_tensor = torch.stack(padded_paragraphs)
    mask_matrices_tensor = torch.stack(mask_matrices)

    return padded_paragraphs_tensor, mask_matrices_tensor

from protocol_encode import protocol2feature, load_sentence_2_vec, get_sentence_embedding
def criteria2embedding(criteria_lst, padding_size):
    sentence2vec = load_sentence_2_vec("data") 
    criteria_lst = [protocol2feature(criteria, sentence2vec) for criteria in criteria_lst] # list of tuple (inclusion_sentence_embedding list, exclusion_sentence_embedding list)

    max_sentences = max(max(p[0].size(0), p[1].size(0)) for p in criteria_lst)
    if max_sentences < padding_size:
        print(f"Warning: padding size is larger than the maximum number of sentences in the data. Padding size: {padding_size}, Max sentences: {max_sentences}")

    incl_criteria = [criteria[0][:padding_size] for criteria in criteria_lst]
    incl_emb, incl_mask = pad_sentences(incl_criteria, padding_size)

    excl_criteria = [criteria[1][:padding_size] for criteria in criteria_lst]
    excl_emb, excl_mask = pad_sentences(excl_criteria, padding_size)

    return incl_emb, incl_mask, excl_emb, excl_mask


def save_train_test_data(train_df, test_df):
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    """
    dense feature: drugs_emb(768) diseases_emb(768) llm_drug_emb(768) llm_disease_emb(768)
    
    incl_emb(32*768), incl_mask(32), excl_emb(32*768), excl_mask(32)
    
    short_onehot(57)
        = gender_vec + phase_vec + min_age_vec + max_age_vec + age_span_vec + inclusion_num_vec, exclusion_num_vec 
        + sentence_num_vec + sentence_ratio_vec +  min max age span + incl excl sentence_num ratio
        
    sparse_feature = country_onehot, state_onehot, city_onehot
    
    """
    

    y_train = train_df['label']
    y_test = test_df['label']
    
    
    print("Saving criteria embeddings...")
    padding_size = 32
    x_train_incl_emb, x_train_incl_mask, x_train_excl_emb, x_train_excl_mask = criteria2embedding(train_df['criteria'], padding_size)
    x_test_incl_emb, x_test_incl_mask, x_test_excl_emb, x_test_excl_mask = criteria2embedding(test_df['criteria'], padding_size)
    
    
    # print(x_train_incl_emb.shape, x_train_incl_mask.shape, x_train_excl_emb.shape, x_train_excl_mask.shape)
    # print(x_test_incl_emb.shape, x_test_incl_mask.shape, x_test_excl_emb.shape, x_test_excl_mask.shape)
    # print(x_test_excl_emb[0], x_test_excl_mask[0])
    # breakpoint()
    
    
    if not os.path.exists('data/data_dcn'):
        os.makedirs('data/data_dcn')
    
    torch.save(x_train_incl_emb, 'data/data_dcn/x_train_inc_emb.pt')
    torch.save(x_train_incl_mask, 'data/data_dcn/x_train_inc_mask.pt')
    torch.save(x_train_excl_emb, 'data/data_dcn/x_train_excl_emb.pt')
    torch.save(x_train_excl_mask, 'data/data_dcn/x_train_excl_mask.pt')
    torch.save(x_test_incl_emb, 'data/data_dcn/x_test_inc_emb.pt')
    torch.save(x_test_incl_mask, 'data/data_dcn/x_test_inc_mask.pt')
    torch.save(x_test_excl_emb, 'data/data_dcn/x_test_excl_emb.pt')
    torch.save(x_test_excl_mask, 'data/data_dcn/x_test_excl_mask.pt')
    
        
    
    trial_df = pd.concat([train_df, test_df], sort=False)
    
    trial_df['age_span'] = np.where((trial_df['max_age'] != -1) & (trial_df['min_age'] != -1),
                                trial_df['max_age'] - trial_df['min_age'], -1)
    trial_df['inclusion_num'] = trial_df['criteria'].apply(lambda x: len(partition_criteria(x)[0]))
    trial_df['exclusion_num'] = trial_df['criteria'].apply(lambda x: len(partition_criteria(x)[1]))
    trial_df['sentence_num'] = trial_df['criteria'].apply(lambda x: len(partition_criteria(x)[0]) + len(partition_criteria(x)[1]))
    trial_df['ratio'] = trial_df['inclusion_num'] / (trial_df['exclusion_num'] + 1)
    

    # Read LLM generated embeddings
    llm_drug_emb_dict = torch.load('llm_emb/data_llm/drug/drug_emb.pt')
    llm_disease_emb_dict = torch.load('llm_emb/data_llm/disease/disease_emb.pt')
    gender_vecs = pickle.load(open('data/phase_vecs.pkl', 'rb'))
    phase_vecs = pickle.load(open('data/phase_vecs.pkl', 'rb'))
    min_age_vecs = pickle.load(open('data/min_age_vecs.pkl', 'rb'))
    max_age_vecs = pickle.load(open('data/max_age_vecs.pkl', 'rb'))
    age_span_vecs = pickle.load(open('data/age_span_vecs.pkl', 'rb'))
    inclusion_num_vecs = pickle.load(open('data/inclusion_num_vecs.pkl', 'rb'))
    exclusion_num_vecs = pickle.load(open('data/exclusion_num_vecs.pkl', 'rb'))
    sentence_num_vecs = pickle.load(open('data/sentence_num_vecs.pkl', 'rb'))
    sentence_ratio_vecs = pickle.load(open('data/ratio_vecs.pkl', 'rb'))
    

    countries_onehot = pickle.load(open('data/countries_onehot.pkl', 'rb'))
    states_onehot = pickle.load(open('data/states_onehot.pkl', 'rb'))
    cities_onehot = pickle.load(open('data/cities_onehot.pkl', 'rb'))
    

    get_sentence_embedding = wrapper_get_sentence_embedding()
    


    dense_feature_list = []
    short_onehot_list = [] 
    country_onehot_list = []
    state_onehot_list = []
    city_onehot_list = []

    
    # Normalize the values
    min_age_min, min_age_max = trial_df['min_age'].min(), trial_df['min_age'].max()
    trial_df['min_age_normalized'] = (trial_df['min_age'] - min_age_min) / (min_age_max - min_age_min)
    max_age_min, max_age_max = trial_df['max_age'].min(), trial_df['max_age'].max()
    trial_df['max_age_normalized'] = (trial_df['max_age'] - max_age_min) / (max_age_max - max_age_min)
    age_span_min, age_span_max = trial_df['age_span'].min(), trial_df['age_span'].max()
    trial_df['age_span_normalized'] = (trial_df['age_span'] - age_span_min) / (age_span_max - age_span_min)
    inclusion_num_min, inclusion_num_max = trial_df['inclusion_num'].min(), trial_df['inclusion_num'].max()
    trial_df['inclusion_num_normalized'] = (trial_df['inclusion_num'] - inclusion_num_min) / (inclusion_num_max - inclusion_num_min)
    exclusion_num_min, exclusion_num_max = trial_df['exclusion_num'].min(), trial_df['exclusion_num'].max()
    trial_df['exclusion_num_normalized'] = (trial_df['exclusion_num'] - exclusion_num_min) / (exclusion_num_max - exclusion_num_min)
    sentence_num_min, sentence_num_max = trial_df['sentence_num'].min(), trial_df['sentence_num'].max()
    trial_df['sentence_num_normalized'] = (trial_df['sentence_num'] - sentence_num_min) / (sentence_num_max - sentence_num_min)
    ratio_min, ratio_max = trial_df['ratio'].min(), trial_df['ratio'].max()
    trial_df['ratio_normalized'] = (trial_df['ratio'] - ratio_min) / (ratio_max - ratio_min)
    
    
    print("Saving other features embeddings...")
    for row_idx, trial_row in tqdm(trial_df.iterrows(), total=len(trial_df)):
        
        
        nctid = trial_row['nctid']
        criteria = trial_row['criteria']
        # Split the string by ';' and remove spaces for each string in the list
        drugs = [drug.strip() for drug in trial_row['drugs'].split(';')]
        diseases = [disease.strip() for disease in trial_row['diseases'].split(';')]

        # dense features
        drugs_emb = torch.mean(torch.stack([get_sentence_embedding(drug) for drug in drugs]), dim=0)
        drugs_llm_emb = torch.mean(torch.stack([llm_drug_emb_dict[drug] for drug in drugs]), dim=0)
        
        diseases_emb = torch.mean(torch.stack([get_sentence_embedding(disease) for disease in diseases]), dim=0)
        diseases_llm_emb = torch.mean(torch.stack([llm_disease_emb_dict[disease] for disease in diseases]), dim=0)
        
        
        gender_vec = torch.tensor(gender_vecs[nctid], dtype=torch.float32)
        phase_vec = torch.tensor(phase_vecs[nctid], dtype=torch.float32)
        min_age_vec = torch.tensor(min_age_vecs[nctid], dtype=torch.float32)
        max_age_vec = torch.tensor(max_age_vecs[nctid], dtype=torch.float32)
        age_span_vec = torch.tensor(age_span_vecs[nctid], dtype=torch.float32)
        inclusion_num_vec = torch.tensor(inclusion_num_vecs[nctid], dtype=torch.float32)
        exclusion_num_vec = torch.tensor(exclusion_num_vecs[nctid], dtype=torch.float32)
        sentence_num_vec = torch.tensor(sentence_num_vecs[nctid], dtype=torch.float32)
        sentence_ratio_vec = torch.tensor(sentence_ratio_vecs[nctid], dtype=torch.float32)
        
        min_age = torch.tensor([trial_row['min_age_normalized']])
        max_age = torch.tensor([trial_row['max_age_normalized']])
        age_span = torch.tensor([trial_row['age_span_normalized']])
        inclusion_num = torch.tensor([trial_row['inclusion_num_normalized']])
        exclusion_num = torch.tensor([trial_row['exclusion_num_normalized']])
        sentence_num = torch.tensor([trial_row['sentence_num_normalized']])
        ratio = torch.tensor([trial_row['ratio_normalized']])

        # sparse features
        country_onehot = torch.tensor(countries_onehot[nctid], dtype=torch.int)
        state_onehot = torch.tensor(states_onehot[nctid], dtype=torch.int)
        city_onehot = torch.tensor(cities_onehot[nctid], dtype=torch.int)

        
        # country, state, city vector add index for none value, add prefix for padding
        # example:  [1, 0, 1, 0, 0] -> [0, 0, 1, 0, 1, 0, 0]
        #           [0, 0, 0, 0, 0] -> [0, 1, 0, 0, 0, 0, 0]
        def process_onehot(onehot_tensor):
            all_zero = torch.all(onehot_tensor == 0)
            if all_zero:
                prefix = torch.tensor([0, 1], dtype=torch.int)
            else:
                prefix = torch.tensor([0, 0], dtype=torch.int)
            processed_tensor = torch.cat((prefix, onehot_tensor), dim=0)
            return processed_tensor
        
        #print idx where there is 1 in state_onehot
        # state_indices = torch.nonzero(state_onehot).squeeze()
        # print(state_indices)
        
        country_onehot = process_onehot(country_onehot)
        state_onehot = process_onehot(state_onehot)
        city_onehot = process_onehot(city_onehot)
        
        # state_indices = torch.nonzero(state_onehot).squeeze()
        # print(state_indices)
        # breakpoint()
           
        # Append all the features to the lists
        dense_feature_list.append(torch.cat((drugs_emb, diseases_emb, drugs_llm_emb, diseases_llm_emb), dim=0))
        
        short_onehot_list.append(torch.cat((gender_vec, phase_vec, min_age_vec, max_age_vec, age_span_vec,
            inclusion_num_vec, exclusion_num_vec, sentence_num_vec, sentence_ratio_vec,
            min_age, max_age, age_span, inclusion_num, exclusion_num, sentence_num, ratio), dim=0))
            
        country_onehot_list.append(country_onehot)
        state_onehot_list.append(state_onehot)
        city_onehot_list.append(city_onehot)



    dense_feature = torch.stack(dense_feature_list)
    short_onehot = torch.stack(short_onehot_list)
    sparse_feature_country = torch.stack(country_onehot_list)
    sparse_feature_state = torch.stack(state_onehot_list)
    sparse_feature_city = torch.stack(city_onehot_list)

    
    #split back to train and test
    x_train_dense = dense_feature[:len(train_df)]
    x_test_dense = dense_feature[len(train_df):]
    x_train_short_onehot = short_onehot[:len(train_df)]
    x_test_short_onehot = short_onehot[len(train_df):]
    x_train_sparse_country = sparse_feature_country[:len(train_df)]
    x_test_sparse_country = sparse_feature_country[len(train_df):]
    x_train_sparse_state = sparse_feature_state[:len(train_df)]
    x_test_sparse_state = sparse_feature_state[len(train_df):]
    x_train_sparse_city = sparse_feature_city[:len(train_df)]
    x_test_sparse_city = sparse_feature_city[len(train_df):]

    torch.save(x_train_dense, 'data/data_dcn/x_train_dense.pt')
    torch.save(x_test_dense, 'data/data_dcn/x_test_dense.pt')
    torch.save(x_train_short_onehot, 'data/data_dcn/x_train_short_onehot.pt')
    torch.save(x_test_short_onehot, 'data/data_dcn/x_test_short_onehot.pt')
    torch.save(x_train_sparse_country, 'data/data_dcn/x_train_sparse_country.pt')
    torch.save(x_test_sparse_country, 'data/data_dcn/x_test_sparse_country.pt')
    torch.save(x_train_sparse_state, 'data/data_dcn/x_train_sparse_state.pt')
    torch.save(x_test_sparse_state, 'data/data_dcn/x_test_sparse_state.pt')
    torch.save(x_train_sparse_city, 'data/data_dcn/x_train_sparse_city.pt')
    torch.save(x_test_sparse_city, 'data/data_dcn/x_test_sparse_city.pt')
    torch.save(y_train, 'data/data_dcn/y_train.pt')
    torch.save(y_test, 'data/data_dcn/y_test.pt')
    



    


 
def load_data():
    
    x_train_inc_emb = torch.load(f'data/data_dcn/x_train_inc_emb.pt')
    x_train_inc_mask = torch.load(f'data/data_dcn/x_train_inc_mask.pt')
    x_train_excl_emb = torch.load(f'data/data_dcn/x_train_excl_emb.pt')
    x_train_excl_mask = torch.load(f'data/data_dcn/x_train_excl_mask.pt')
    x_test_inc_emb = torch.load(f'data/data_dcn/x_test_inc_emb.pt')
    x_test_inc_mask = torch.load(f'data/data_dcn/x_test_inc_mask.pt')
    x_test_excl_emb = torch.load(f'data/data_dcn/x_test_excl_emb.pt')
    x_test_excl_mask = torch.load(f'data/data_dcn/x_test_excl_mask.pt')
    
    x_train_dense = torch.load(f'data/data_dcn/x_train_dense.pt')
    x_test_dense = torch.load(f'data/data_dcn/x_test_dense.pt')
    
    x_train_short_onehot = torch.load(f'data/data_dcn/x_train_short_onehot.pt')
    x_test_short_onehot = torch.load(f'data/data_dcn/x_test_short_onehot.pt')
    
    
    x_train_sparse_country = torch.load(f'data/data_dcn/x_train_sparse_country.pt')
    x_test_sparse_country = torch.load(f'data/data_dcn/x_test_sparse_country.pt')
    x_train_sparse_state = torch.load(f'data/data_dcn/x_train_sparse_state.pt')
    x_test_sparse_state = torch.load(f'data/data_dcn/x_test_sparse_state.pt')
    x_train_sparse_city = torch.load(f'data/data_dcn/x_train_sparse_city.pt')
    x_test_sparse_city = torch.load(f'data/data_dcn/x_test_sparse_city.pt')
    y_train = torch.load(f'data/data_dcn/y_train.pt')
    y_test = torch.load(f'data/data_dcn/y_test.pt')
    
    
    
    return x_train_inc_emb, x_train_inc_mask, x_train_excl_emb, x_train_excl_mask, x_test_inc_emb,\
        x_test_inc_mask, x_test_excl_emb, x_test_excl_mask, x_train_dense, x_test_dense, x_train_short_onehot,\
            x_test_short_onehot, x_train_sparse_country, x_test_sparse_country, x_train_sparse_state, x_test_sparse_state,\
                x_train_sparse_city, x_test_sparse_city, y_train, y_test
    


if __name__ == '__main__':
    

    train_df = pd.read_csv(f'../data_llm/enrollment_timefiltered_train.csv', sep='\t')
    test_df = pd.read_csv(f'../data_llm/enrollment_timefiltered_test.csv', sep='\t')
    
    save_train_test_data(train_df, test_df)
    
    
