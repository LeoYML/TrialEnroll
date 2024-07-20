from tqdm import tqdm 
from xml.etree import ElementTree as ET
from datetime import datetime
import re
import pandas as pd
import pickle

"""
    age_dict = { nct_id: [min_age, max_age, span] }

"""

def save_dict(file, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(file, f)

def load_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def parse_age(age_text):
    
    if age_text == '' or age_text == 'N/A':
        return -1
    age = int(age_text.split()[0])
    return age

def xmlfile2age(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    min_age_elem = root.find('.//minimum_age')
    max_age_elem = root.find('.//maximum_age')
    
    min_age = min_age_elem.text if (min_age_elem is not None and min_age_elem != "N/A") else ''
    max_age = max_age_elem.text if (max_age_elem is not None and max_age_elem != "N/A") else ''
    
    
    # min_age = parse_age(min_age)
    # max_age = parse_age(max_age)

    return min_age, max_age


def get_age():

    
    
    age_dict = {}

    # 478504 lines
    with open("../data/trials/all_xml.txt", "r") as file:
        for xml_path in tqdm(file):
            xml_path = f"../data/{xml_path.strip()}"

            # NCT00000150 <- raw_data/NCT0000xxxx/NCT00000150.xml
            nct_id = re.search(r"/([^/]+)\.xml$", xml_path).group(1)
            
            min_age, max_age = xmlfile2age(xml_path)
            
            #print(min_age, max_age)
            
            min_age = parse_age(min_age)
            max_age = parse_age(max_age)
            
            if min_age != -1 and max_age != -1:
                span = max_age - min_age
            else:
                span = -1
            
            #print(min_age, max_age, span, "\n")
                
            age_dict[nct_id] = [min_age, max_age, span]
    
    print(age_dict['NCT00000102'])
    print(len(age_dict))
    
    save_dict(age_dict, '../data/age_dict.pkl')
    
    return age_dict


if __name__ == "__main__":
    get_age()

    # age_dict = load_dict('data/age_dict.pkl')
    # print(age_dict['NCT00000102'])
    # print(len(age_dict))