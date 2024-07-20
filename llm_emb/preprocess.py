import os
import sys
sys.path.append(os.getcwd())


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import time
import pandas as pd
import random
from tqdm import tqdm
import traceback
import json


def get_drug_disease_list():
    train_df = pd.read_csv("../data/enrollment_timefiltered_train.csv", sep='\t')
    test_df = pd.read_csv("../data/enrollment_timefiltered_test.csv", sep='\t')
    
    drugs = []
    diseases = []
    for index, row in train_df.iterrows():
        drugs.extend(row['drugs'].split(";"))
        diseases.extend(row['diseases'].split(";"))
    for index, row in test_df.iterrows():
        drugs.extend(row['drugs'].split(";"))
        diseases.extend(row['diseases'].split(";"))
    
    unique_drugs = list(set(drugs))
    unique_diseases = list(set(diseases))

    # save and read
    
    with open('data_llm/drug/drugs.txt', 'w') as file:
        for drug in unique_drugs:
            file.write(drug + "\n")
    with open('data_llm/disease/diseases.txt', 'w') as file:
        for disease in unique_diseases:
            file.write(disease + "\n")
    
    
    
    print(f"Number of drugs: {len(unique_drugs)}")
    print(f"Number of diseases: {len(unique_diseases)}")
    
    return drugs


def read_drug_list():
    with open('data_llm/drug/drugs.txt', 'r') as file:
        drugs = file.readlines()
        drugs = [drug.strip() for drug in drugs]
    return drugs

def read_disease_list():
    with open('data_llm/disease/diseases.txt', 'r') as file:
        diseases = file.readlines()
        diseases = [diseases.strip() for diseases in diseases]
    return diseases

if __name__ == "__main__":
    
    get_drug_disease_list()
    drugs = read_drug_list()
    print(len(drugs))
    diseases = read_disease_list()
    print(len(diseases))
    
    #check()
    #read_drug_list()
    #read_disease_list()