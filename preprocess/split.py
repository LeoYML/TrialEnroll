import pandas as pd
import os
import torch
import sys



# Ensure the helper functions are accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from helper import load_data

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_num_threads(16)


# # Load the x y data
# x_data, y_data = load_data()
# import collections
# print(collections.Counter(y_data.numpy()))

def data_split():
    # load raw
    trial_df = pd.read_csv(f'../data/trial_data.csv', sep='\t')

    # add start_year and completion_year columns
    # example 'start_year' = "July 2002"
    def extract_year(date_str):
        if isinstance(date_str, str):
            return date_str.split()[-1]
        return None

    # Apply the function to extract years from 'start_date' and 'completion_date' columns
    trial_df['start_year'] = trial_df['start_date'].apply(extract_year)
    trial_df['completion_year'] = trial_df['completion_date'].apply(extract_year)

    # num of rows
    print("all:", trial_df.shape[0])
    # valid rows
    valid_list = []
    for index, row in trial_df.iterrows():
        if row['start_year'] is not None and row['completion_year'] is not None:
            valid_list.append(row)
            
    print("valid:", len(valid_list))
    #filter out rows according to valid_list
    trial_df = pd.DataFrame(valid_list)


    train_df = trial_df[trial_df['completion_year'] < '2015']
    test_df = trial_df[trial_df['start_year'] >= '2015']
    
    # Step 3: Count the number of occurrences of each label in train_df
    train_label_counts = train_df['label'].value_counts()

    # Step 4: Count the number of occurrences of each label in test_df
    test_label_counts = test_df['label'].value_counts()
    print("Label counts in train_df:")
    print(train_label_counts)

    print("\nLabel counts in test_df:")
    print(test_label_counts)

    print(len(train_df)/len(test_df))
    print(len(train_df), len(test_df), len(train_df)+len(test_df))
    
    train_nctids = set(train_df['nctid'])
    test_nctids = set(test_df['nctid'])

    common_nctids = train_nctids.intersection(test_nctids)

    if common_nctids:
        print("Common 'nctid' found:", common_nctids)
    else:
        print("No common 'nctid' found.")
        
    
    
    train_df.to_csv("../data/enrollment_timefiltered_train.csv", sep='\t', index=False)
    test_df.to_csv("../data/enrollment_timefiltered_test.csv", sep='\t', index=False)
    
    train_idx_list = trial_df[trial_df['nctid'].isin(train_df['nctid'])].index.tolist()
    test_idx_list = trial_df[trial_df['nctid'].isin(test_df['nctid'])].index.tolist()
    
    common_indices = set(train_idx_list).intersection(test_idx_list)

    if common_indices:
        print("Common indices found:", common_indices)
    else:
        print("No common indices found.")


    # Print the indices
    print("Train Indices:", len(train_idx_list))
    print("Test Indices:", len(test_idx_list))
    
    with open('../data/train_idx.txt', 'w') as f:
        for idx in train_idx_list:
            f.write(f"{idx}\n")

    with open('../data/test_idx.txt', 'w') as f:
        for idx in test_idx_list:
            f.write(f"{idx}\n")
    


    
def read_indices(file_path):
    with open(file_path, 'r') as f:
        indices = [int(line.strip()) for line in f]
    return indices


    

    
if __name__ == "__main__":

    data_split()
    
    train_df = pd.read_csv('../data/enrollment_timefiltered_train.csv', sep='\t')
    test_df = pd.read_csv('../data/enrollment_timefiltered_test.csv', sep='\t')
    
    print("Train:", len(train_df), "Test:", len(test_df))
    
    # train_idx_list = read_indices('../data/train_idx.txt')
    # test_idx_list = read_indices('../data/test_idx.txt')

    #test()

    # Check for common values using set intersection
    
