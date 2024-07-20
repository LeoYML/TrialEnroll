
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


os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def save_all_batches(model, data_name):
    directory = f'data_llm/{data_name}/{model}'
    all_results = {}
    total_num_data = 0

    batch_files = [f for f in os.listdir(directory) if f.endswith('.json')]

    for batch_file in batch_files:
        batch_file_path = os.path.join(directory, batch_file)
        
        with open(batch_file_path, 'r') as f:
            batch_results = json.load(f)
        
        all_results.update(batch_results)
        
        # Count the number of data during iterations
        num_data = len(all_results)
        total_num_data += num_data
        print(f"Number of data: {num_data}")
        
    # Count the total number of data through iterations
    #print(f"Total number of data: {total_num_data}")
    
    # Save all results
    all_results_file = f'data_llm/{data_name}/{data_name}_llm_generated.json'
    with open(all_results_file, 'w') as f:
        json.dump(all_results, f, indent=4)

        
    return all_results

def read_llm_generated(model, data_name):
    all_results_file = f'data_llm/{data_name}/{data_name}_llm_generated.json'
    with open(all_results_file, 'r') as f:
        all_results = json.load(f)
    
    return all_results


from preprocess import get_drug_disease_list, read_drug_list, read_disease_list
def check():
    
    drug_list = read_drug_list()
    drug_llm_generated = read_llm_generated("MistralInstruct", "drug")
    print(len(drug_list), len(drug_llm_generated))

    excluded_drugs = [drug for drug in drug_llm_generated if drug not in drug_list]
    print(excluded_drugs)
    excluded_drugs = [drug for drug in drug_list if drug not in drug_llm_generated]
    print(excluded_drugs)
    
    disease_list = read_disease_list()
    disease_llm_generated = read_llm_generated("MistralInstruct", "disease")
    print(len(disease_list), len(disease_llm_generated))
    
    excluded_diseases = [disease for disease in disease_llm_generated if disease not in disease_list]
    print(excluded_diseases)
    excluded_diseases = [disease for disease in disease_list if disease not in disease_llm_generated]
    print(excluded_diseases)


from transformers import AutoTokenizer, AutoModel
def wrapper_get_sentence_embedding():
    model_name = "dmis-lab/biobert-base-cased-v1.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name).to(device)

    def get_sentence_embedding(sentence):
        # Encode the input string
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        
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


def get_drug_llm_embedding():
    
    get_sentence_embedding = wrapper_get_sentence_embedding()
    drug_llm_generated = read_llm_generated("MistralInstruct", "drug")
    
    drug_emb_dict = {}
    

    #drug_llm_generated = {k: drug_llm_generated[k] for k in list(drug_llm_generated)[:10]}
    
    
    for drug in tqdm(drug_llm_generated):
        drug_emb = get_sentence_embedding(drug)
        drug_emb_dict[drug] = drug_emb
    
    torch.save(drug_emb_dict, 'data_llm/drug/drug_llm_emb.pt')
    
def get_disease_llm_embedding():
    
    get_sentence_embedding = wrapper_get_sentence_embedding()
    disease_llm_generated = read_llm_generated("MistralInstruct", "disease")
    
    disease_emb_dict = {}
    

    
    
    for disease in tqdm(disease_llm_generated):
        disease_emb = get_sentence_embedding(disease)
        disease_emb_dict[disease] = disease_emb
    
    torch.save(disease_emb_dict, 'data_llm/disease/disease_llm_emb.pt')
    
    
def read_drug_llm_emb(path = 'data_llm/drug/drug_emb.pt'):
    drug_emb_dict = torch.load(path)
    return drug_emb_dict

def read_disease_llm_emb(path = 'data_llm/disease/disease_emb.pt'):
    disease_emb_dict = torch.load(path)
    return disease_emb_dict
    

def save_embeddings():
    
    save_all_batches("MistralInstruct", "drug")
    get_drug_llm_embedding()
    
    save_all_batches("MistralInstruct", "disease")
    get_disease_llm_embedding()


if __name__ == "__main__":


    # get_drug_llm_embedding()
    
    # drug_emb = read_drug_llm_emb()
    # print(drug_emb.keys())
    # print(drug_emb["placebo, oral intake single dose"])
    
    #save_all_batches("MistralInstruct", "disease")
    # disease = read_llm_generated("MistralInstruct", "disease")
    # print(len(disease))
    
    check()
    save_embeddings()
    