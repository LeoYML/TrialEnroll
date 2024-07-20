## versions
python 3.10, torch 2.3 

## ClinicalTrial.gov
For more information and resources related to HetioNet, visit the:
[ClinicalTrial.gov](https://clinicaltrials.gov)

### Getting Started
1. **Download the ClinicalTrial Data**  
```
    cd data
    wget https://clinicaltrials.gov/AllPublicXML.zip
```
2. **Decompress the Data File**  
```
    unzip AllPublicXML.zip -d trials
    find trials/* -name "NCT*.xml" | sort > trials/all_xml.txt
```
3. **IQVIA Label Data**
Download the IQVIA label to data/:
https://github.com/futianfan/clinical-trial-outcome-prediction/tree/main/IQVIA

Rename the file name trial_outcomes_v1.csv to IQVIA_trial_outcomes.csv.


### df preprocessing
cd TrialEnroll/preprocess
run collect_age.py, collect_location.py, collect_str.py, collect_time.py
run save_df.py

### LLM generated features preprocessing
huggingface-cli download --resume-download mistralai/Mistral-7B-Instruct-v0.3 --local-dir 7B-Instruct-v0.3
set mistral path in llm_request_MistralInstruct.py
cd TrialEnroll/llm_emb
create folder data_llm/disease/MistralInstruct, data_llm/drug/MistralInstruct
run preprocess.py
run llm_request_MistralInstruct.py
run embedding.py

### Prepare Criteria embedding
cd TrialEnroll
run protocol_encode.py

### run DCN
run col_preprocessing.py
run stack_features_dcn.py
run hatten_cross.py

PR AUC: 0.7015