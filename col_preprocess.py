import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from collections import defaultdict

import random
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_num_threads(16)

random.seed(0)

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


    

def preprocess():
    
    train_df = pd.read_csv(f'data/enrollment_timefiltered_train.csv', sep='\t')
    test_df = pd.read_csv(f'data/enrollment_timefiltered_test.csv', sep='\t')
    
    trial_df = pd.concat([train_df, test_df], sort=False)
    
    trial_df['age_span'] = np.where((trial_df['max_age'] != -1) & (trial_df['min_age'] != -1),
                                trial_df['max_age'] - trial_df['min_age'], -1)
    trial_df['inclusion_num'] = trial_df['criteria'].apply(lambda x: len(partition_criteria(x)[0]))
    trial_df['exclusion_num'] = trial_df['criteria'].apply(lambda x: len(partition_criteria(x)[1]))
    trial_df['sentence_num'] = trial_df['criteria'].apply(lambda x: len(partition_criteria(x)[0]) + len(partition_criteria(x)[1]))
    trial_df['ratio'] = trial_df['inclusion_num'] / (trial_df['exclusion_num'] + 1)
    
    
    

    
    
    features = ['min_age', 'max_age', 'age_span', 'inclusion_num', 'exclusion_num', 'sentence_num', 'ratio']
    
    

    
    drug_dict = {}
    disease_dict = {}
    country_dict = {}
    state_dict = {}
    city_dict = {}
    
    def process_column_to_list(value):
        return [item.strip() for item in value.split(";")]
    
    for idx, row in trial_df.iterrows():
        
        drugs = [drug.strip() for drug in row['drugs'].split(';')]
        diseases = [disease.strip() for disease in row['diseases'].split(';')]
        
        drug_dict[row['nctid']] = drugs
        disease_dict[row['nctid']] = diseases
    
    location_dict = load_dict('data/location_dict.pkl')
    
    for nctid, location in location_dict.items():
        country_dict[nctid] = location.get('countries', [])
        state_dict[nctid] = location.get('states', [])
        city_dict[nctid] = location.get('cities', [])
    
    
    # save the dictionaries to json
    json.dump(drug_dict, open('data/drug_dict.json', 'w'))
    json.dump(disease_dict, open('data/disease_dict.json', 'w'))
    json.dump(country_dict, open('data/country_dict.json', 'w'))
    json.dump(state_dict, open('data/state_dict.json', 'w'))
    json.dump(city_dict, open('data/city_dict.json', 'w'))
    
    
   

    # for feature in features:
    #     plt.figure(figsize=(10, 6))
    #     sns.histplot(trial_df[feature], kde=True)
    #     plt.title(f'Distribution of {feature}')
    #     plt.xlabel(feature)
    #     plt.ylabel('Frequency')


    #     plt.savefig(f'plots/distribution_of_{feature}.png')
    #     plt.close()  

    
    # print(trial_df[features].describe())
       

    
    # split train and test back
    train_df = trial_df.iloc[:len(train_df)]
    test_df = trial_df.iloc[len(train_df):]
    
    # # save train and test
    train_df.to_csv('data/enrollment_timefiltered_train.csv', sep='\t', index=False)
    test_df.to_csv('data/enrollment_timefiltered_test.csv', sep='\t', index=False)
    

def generate_onehot_encode(item_dict):
    
    unique_items = sorted(set(item for items in item_dict.values() for item in items))
    item_to_index = {item: idx for idx, item in enumerate(unique_items)}
    items_onehot = defaultdict(list)
    for key, drugs in item_dict.items():
        one_hot_vector = [0] * len(unique_items)
        for drug in drugs:
            if drug in item_to_index:
                one_hot_vector[item_to_index[drug]] = 1
        items_onehot[key] = one_hot_vector
    return items_onehot
 
    
def save_onehot_encode():
    
    train_df = pd.read_csv('data/enrollment_timefiltered_train.csv', sep='\t')
    test_df = pd.read_csv('data/enrollment_timefiltered_test.csv', sep='\t')
    trial_df = pd.concat([train_df, test_df], sort=False)
    
    print(f'train_df: {len(train_df)}, test_df: {len(test_df)}, trial_df: {len(trial_df)}')
    
    drug_dict = json.load(open('data/drug_dict.json', 'r'))
    disease_dict = json.load(open('data/disease_dict.json', 'r'))
    country_dict = json.load(open('data/country_dict.json', 'r'))
    state_dict = json.load(open('data/state_dict.json', 'r'))
    city_dict = json.load(open('data/city_dict.json', 'r'))
    
    nctids = set(trial_df['nctid'])
    country_dict = {k: v for k, v in country_dict.items() if k in nctids}
    state_dict = {k: v for k, v in state_dict.items() if k in nctids}
    city_dict = {k: v for k, v in city_dict.items() if k in nctids}
    json.dump(country_dict, open('data/country_dict.json', 'w'))
    json.dump(state_dict, open('data/state_dict.json', 'w'))
    json.dump(city_dict, open('data/city_dict.json', 'w'))

    
    print("Generating one-hot encoding for drugs...")
    drugs_onehot = generate_onehot_encode(drug_dict)
    with open('data/drugs_onehot.pkl', 'wb') as f:
        pickle.dump(drugs_onehot, f)
    print(f"drugs_onehot: {len(drugs_onehot)}, length of one-hot vec: {len(drugs_onehot[list(drugs_onehot.keys())[0]])}")  

    print("Generating one-hot encoding for diseases...")
    diseases_onehot = generate_onehot_encode(disease_dict)
    with open('data/diseases_onehot.pkl', 'wb') as f:
        pickle.dump(diseases_onehot, f)
    print(f"diseases_onehot: {len(diseases_onehot)}, length of one-hot vec: {len(diseases_onehot[list(diseases_onehot.keys())[0]])}")
    
    print("Generating one-hot encoding for countries...")
    countries_onehot = generate_onehot_encode(country_dict)
    with open('data/countries_onehot.pkl', 'wb') as f:
        pickle.dump(countries_onehot, f)
    print(f"countries_onehot: {len(countries_onehot)}, length of one-hot vec: {len(countries_onehot[list(countries_onehot.keys())[0]])}")
    
    print("Generating one-hot encoding for states...")
    states_onehot = generate_onehot_encode(state_dict)
    with open('data/states_onehot.pkl', 'wb') as f:
        pickle.dump(states_onehot, f)
    print(f"states_onehot: {len(states_onehot)}, length of one-hot vec: {len(states_onehot[list(states_onehot.keys())[0]])}")
    
    print("Generating one-hot encoding for cities...")
    cities_onehot = generate_onehot_encode(city_dict)
    with open('data/cities_onehot.pkl', 'wb') as f:
        pickle.dump(cities_onehot, f)   
    print(f"cities_onehot: {len(cities_onehot)}, length of one-hot vec: {len(cities_onehot[list(cities_onehot.keys())[0]])}")

    
    # train_df = trial_df.iloc[:len(train_df)]
    # test_df = trial_df.iloc[len(train_df):]
    
    # train_df.to_csv('data/enrollment_timefiltered_train.csv', sep='\t', index=False)
    # test_df.to_csv('data/enrollment_timefiltered_test.csv', sep='\t', index=False)

def generate_bin_vector( df, col_name):
    
    sentence_num_vecs = {}
    
    quartiles = df[col_name].quantile([0.25, 0.5, 0.75])
    
    def get_bin_vector(value, quartiles):
        if value <= quartiles[0.25]:
            return [1, 0, 0, 0]
        elif value <= quartiles[0.5]:
            return [0, 1, 0, 0]
        elif value <= quartiles[0.75]:
            return [0, 0, 1, 0]
        else:
            return [0, 0, 0, 1]
    
    # Iterate over the rows and assign binary vectors
    for index, row in df.iterrows():
        nctid = row['nctid']
        sentence_num = row[col_name]
        bin_vector = get_bin_vector(sentence_num, quartiles)
        sentence_num_vecs[nctid] = bin_vector
        
    return sentence_num_vecs

   
def save_bin_vector():
    
    train_df = pd.read_csv('data/enrollment_timefiltered_train.csv', sep='\t')
    test_df = pd.read_csv('data/enrollment_timefiltered_test.csv', sep='\t')
    trial_df = pd.concat([train_df, test_df], sort=False)
    
    sentence_num_vecs = generate_bin_vector(trial_df, 'sentence_num')
    
    inclusion_num_vecs = generate_bin_vector(trial_df, 'inclusion_num')
    exclusion_num_vecs = generate_bin_vector(trial_df, 'exclusion_num')
    ratio_vecs = generate_bin_vector(trial_df, 'ratio')
    
    # print(ratio_vecs["NCT00000105"],"##", trial_df[trial_df['nctid'] == "NCT00000105"]['ratio'], trial_df[trial_df['nctid'] == "NCT00000105"]['inclusion_num'], trial_df[trial_df['nctid'] == "NCT00000105"]['exclusion_num'])
    # print(ratio_vecs["NCT00000173"],"##", trial_df[trial_df['nctid'] == "NCT00000173"]['ratio'], trial_df[trial_df['nctid'] == "NCT00000173"]['inclusion_num'], trial_df[trial_df['nctid'] == "NCT00000173"]['exclusion_num'])
    # print(ratio_vecs["NCT00002545"],"##", trial_df[trial_df['nctid'] == "NCT00002545"]['ratio'], trial_df[trial_df['nctid'] == "NCT00002545"]['inclusion_num'], trial_df[trial_df['nctid'] == "NCT00002545"]['exclusion_num'])
    
    
    unique_genders = trial_df['gender'].unique()
    gender_dict = {gender: [1 if i == idx else 0 for i in range(len(unique_genders))]
                   for idx, gender in enumerate(unique_genders)}
    unique_phases = trial_df['phase'].unique()
    phase_dict = {phase: [1 if i == idx else 0 for i in range(len(unique_phases))]
                   for idx, phase in enumerate(unique_phases)}    

    gender_vecs = {}
    phase_vecs = {}
    
    # Assign binary vectors for 'gender'
    for index, row in trial_df.iterrows():
        nctid = row['nctid']
        gender = row['gender']
        gender_vec = gender_dict[gender]
        gender_vecs[nctid] = gender_vec
        
        phase = row['phase']
        phase_vec = phase_dict[phase]
        phase_vecs[nctid] = phase_vec
    
    
    def min_age_bucket(min_age):
        return (
            [1, 0, 0, 0, 0, 0] if min_age == -1 else
            [0, 1, 0, 0, 0, 0] if 0 <= min_age <= 17 else
            [0, 0, 1, 0, 0, 0] if min_age == 18 else
            [0, 0, 0, 1, 0, 0] if 19 <= min_age <= 39 else
            [0, 0, 0, 0, 1, 0] if 40 <= min_age <= 59 else
            [0, 0, 0, 0, 0, 1] if min_age >= 60 else
            [0, 0, 0, 0, 0, 0]
        )
    min_age_vecs = {row['nctid']: min_age_bucket(row['min_age']) for _, row in trial_df.iterrows()}
    

    
    def max_age_bucket(max_age):
        return (
            [1, 0, 0, 0, 0, 0] if max_age == -1 else
            [0, 1, 0, 0, 0, 0] if 0 <= max_age <= 18 else
            [0, 0, 1, 0, 0, 0] if 19 <= max_age <= 39 else
            [0, 0, 0, 1, 0, 0] if 40 <= max_age <= 59 else
            [0, 0, 0, 0, 1, 0] if 60 <= max_age <= 79 else
            [0, 0, 0, 0, 0, 1] if max_age >= 80 else
            [0, 0, 0, 0, 0, 0]
        )
    max_age_vecs = {row['nctid']: max_age_bucket(row['max_age']) for _, row in trial_df.iterrows()}
    
    def age_span_bucket(age_span):
        return (
            [1, 0, 0, 0, 0, 0] if age_span == -1 else
            [0, 1, 0, 0, 0, 0] if 0 <= age_span <= 19 else
            [0, 0, 1, 0, 0, 0] if 20 <= age_span <= 39 else
            [0, 0, 0, 1, 0, 0] if 40 <= age_span <= 59 else
            [0, 0, 0, 0, 1, 0] if 60 <= age_span <= 79 else
            [0, 0, 0, 0, 0, 1] if age_span >= 80 else
            [0, 0, 0, 0, 0, 0]
        )

    age_span_vecs = {row['nctid']: age_span_bucket(row['age_span']) for _, row in trial_df.iterrows()}
    # print(age_span_vecs["NCT00000105"], trial_df[trial_df['nctid'] == "NCT00000105"]["age_span"])
    # print(age_span_vecs["NCT00000173"], trial_df[trial_df['nctid'] == "NCT00000173"]["age_span"])
    # print(age_span_vecs["NCT00002545"], trial_df[trial_df['nctid'] == "NCT00002545"]["age_span"])
    # print(age_span_vecs["NCT00003105"], trial_df[trial_df['nctid'] == "NCT00003105"]["age_span"])
    # print(age_span_vecs["NCT00003256"], trial_df[trial_df['nctid'] == "NCT00003256"]["age_span"])
    
    # save the vectors
    with open('data/sentence_num_vecs.pkl', 'wb') as f:
        pickle.dump(sentence_num_vecs, f)
    with open('data/inclusion_num_vecs.pkl', 'wb') as f:
        pickle.dump(inclusion_num_vecs, f)
    with open('data/exclusion_num_vecs.pkl', 'wb') as f:
        pickle.dump(exclusion_num_vecs, f) 
    with open('data/ratio_vecs.pkl', 'wb') as f:
        pickle.dump(ratio_vecs, f)
    with open('data/gender_vecs.pkl', 'wb') as f:
        pickle.dump(gender_vecs, f)
    with open('data/phase_vecs.pkl', 'wb') as f:
        pickle.dump(phase_vecs, f)
    with open('data/min_age_vecs.pkl', 'wb') as f:
        pickle.dump(min_age_vecs, f)
    with open('data/max_age_vecs.pkl', 'wb') as f:
        pickle.dump(max_age_vecs, f)
    with open('data/age_span_vecs.pkl', 'wb') as f:
        pickle.dump(age_span_vecs, f)
        
    print(f"sentence_num_vecs: {len(sentence_num_vecs)}, length of bin vec: {len(sentence_num_vecs[list(sentence_num_vecs.keys())[0]])}")
    print(f"inclusion_num_vecs: {len(inclusion_num_vecs)}, length of bin vec: {len(inclusion_num_vecs[list(inclusion_num_vecs.keys())[0]])}")
    print(f"exclusion_num_vecs: {len(exclusion_num_vecs)}, length of bin vec: {len(exclusion_num_vecs[list(exclusion_num_vecs.keys())[0]])}")
    print(f"ratio_vecs: {len(ratio_vecs)}, length of bin vec: {len(ratio_vecs[list(ratio_vecs.keys())[0]])}")
    print(f"genders_vecs: {len(gender_vecs)}, length of bin vec: {len(gender_vecs[list(ratio_vecs.keys())[0]])}")
    print(f"phase_vecs: {len(phase_vecs)}, length of bin vec: {len(phase_vecs[list(ratio_vecs.keys())[0]])}")
    print(f"min_age_vecs: {len(min_age_vecs)}, length of bin vec: {len(min_age_vecs[list(ratio_vecs.keys())[0]])}")
    print(f"max_age_vecs: {len(max_age_vecs)}, length of bin vec: {len(max_age_vecs[list(ratio_vecs.keys())[0]])}")
    print(f"age_span_vecs: {len(age_span_vecs)}, length of bin vec: {len(age_span_vecs[list(ratio_vecs.keys())[0]])}")
    
    
    
if __name__ == '__main__':

    preprocess()
    save_onehot_encode()
    save_bin_vector()
    
    
    