

mistral_models_path = """your mistral path""


# snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)


# %%
from mistral_inference.model import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

import pandas as pd


tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path)


# %%
def llm_request_MistralInstruct(system_prompt, user_prompt, model_name='mistral'):
    if model_name == 'mistral':
        mess_list = [SystemMessage(content=system_prompt),
                      UserMessage(content=user_prompt)]
        completion_request = ChatCompletionRequest(messages=mess_list)
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        out_tokens, _ = generate([tokens], model, max_tokens=2048, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

        return result
    else:
        raise ValueError(f"Model {model} not supported")


from tqdm import tqdm

# drugs = ["isentress\u00ae (raltegravir, 400 every 12 hours)", "candesartan + chlorthalidone (8 mg + 25 mg)", "placebo", "lfa102"]

# system_prompt = """
# You are a highly knowledgeable clinical pharmacologist. Given a string that contains the name of a drug, please:

# 1. Provide the name of the drug.
# 2. Offer a comprehensive description of the drug, including:
#    - Mechanism of action
#    - Common uses
#    - Notable side effects
# 3. Discuss the difficulty of recruiting patients for clinical trials involving this drug, including the reasons behind these challenges.

# Noted instruction: Please respond with fewer than 100 words.
# """

# for i, drug_name in tqdm(enumerate(drugs)):
#     user_prompt = f"The following string contains the name of a drug: <string>{drug_name}</string>"

#     result = llm_request_MistralInstruct(system_prompt, user_prompt)

#     print(f"{result}\n========================\n")

from preprocess import get_drug_disease_list, read_drug_list, read_disease_list
import json


def drug_prompt(drug, model):
    
    # GPT request
    if model == "MistralInstruct":
        system_prompt = """
            You are a highly knowledgeable clinical pharmacologist. Given a string that contains the name of a drug, please:

            1. Provide the name of the drug.
            2. Offer a comprehensive description of the drug, including:
            - Mechanism of action
            - Common uses
            - Notable side effects
            3. Discuss the difficulty of recruiting patients for clinical trials involving this drug, including the reasons behind these challenges.

            Noted instruction: Please respond with fewer than 100 words.
        """
        
        user_prompt = f"The following string contains the name of a drug: <string>{drug}</string>"

        result = llm_request_MistralInstruct(system_prompt, user_prompt)
        
        return result

        
    
    else:
        raise ValueError("Invalid model")
    

def disease_prompt(disease, model):
    
    # GPT request
    if model == "MistralInstruct":
        system_prompt = """
            You are a highly knowledgeable clinical epidemiologist. Given a string that contains the name of a disease, please:

            1. Provide the name of the disease.
            2. Offer a comprehensive description of the disease, including:
            - Pathogenesis (mechanism of disease development)
            - Common symptoms
            - Typical treatment options
            3. Discuss the difficulty of recruiting patients for clinical trials involving this disease, including the reasons behind these challenges.

            Noted instruction: Please respond with fewer than 100 words.
        """
        
        user_prompt = f"The following string contains the name of a disease: <string>{disease}</string>"

        result = llm_request_MistralInstruct(system_prompt, user_prompt)
        
        return result
 


def save_drug_result(model = "MistralInstruct"):
    
    drugs = read_drug_list()
    
    print(f"Size of drugs: {len(drugs)}")
    
    #drugs = drugs[:10]
    
    batch_size = 50
    
    drug_llm_results = {}

    #os.makedirs(f'data_llm/drug/{model}', exist_ok=True)

    # add dqtm 

    
    for i in tqdm(range(0, len(drugs), batch_size), desc="Processing batches"):
        
        batch = drugs[i:i + batch_size]
        batch_results = {}

        for drug in batch:
            
            
            try:
                llm_result = drug_prompt(drug, model)
                batch_results[drug] = llm_result
                #time.sleep(2) 
            except Exception as e:
                print(f"Error processing {drug}: {e}")
                
        
        batch_file_name = f'data_llm/drug/{model}/drug_{model}_batch_{i // batch_size + 1}.json'
        with open(batch_file_name, 'w') as f:
            json.dump(batch_results, f, indent=4)

        drug_llm_results.update(batch_results)

        print(f"Batch {i // batch_size + 1} saved to {batch_file_name}")
        
        

def save_disease_result(model = "MistralInstruct"):
    
    diseases = read_disease_list()
    
    print(f"Size of diseases: {len(diseases)}")
    
    #diseases = diseases[:10]
    
    batch_size = 50
    
    disease_llm_results = {}


    # add dqtm 

    
    for i in tqdm(range(0, len(diseases), batch_size), desc="Processing batches"):
        
        batch = diseases[i:i + batch_size]
        batch_results = {}

        for disease in batch:
            
            
            try:
                llm_result = disease_prompt(disease, model)
                batch_results[disease] = llm_result
                #time.sleep(2) 
            except Exception as e:
                print(f"Error processing {disease}: {e}")
                
        
        batch_file_name = f'data_llm/disease/{model}/disease_{model}_batch_{i // batch_size + 1}.json'
        with open(batch_file_name, 'w') as f:
            json.dump(batch_results, f, indent=4)

        disease_llm_results.update(batch_results)

        print(f"Batch {i // batch_size + 1} saved to {batch_file_name}")
     




if __name__ == "__main__":
    
    save_drug_result(model="MistralInstruct")
    save_disease_result(model="MistralInstruct")
    

    
