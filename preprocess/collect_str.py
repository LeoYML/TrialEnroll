from tqdm import tqdm 
from xml.etree import ElementTree as ET
from datetime import datetime
import re
import pandas as pd
import pickle

"""
    str_dict = {
        "gender": gender,
        "phase": phase,
        "locations": locations,
        "condition": condition,
        "intervention": intervention
    }

"""

def save_dict(file, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(file, f)

def load_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def xmlfile2str(xml_file):
    
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    gender_elem = root.find('.//gender')
    gender = gender_elem.text if (gender_elem is not None) else 'N/A'
    
    phase_elem = root.find('.//phase')
    phase = phase_elem.text if (phase_elem is not None) else 'N/A'
    
    location_names = root.findall(".//location/facility/name")
    locations = [name.text for name in location_names]
    
    # Disease
    condition = root.findtext(".//condition")  
    
    # Drug, Procedure, Biological, etc
    intervention_name = root.findtext(".//intervention/intervention_name") if root.findtext(".//intervention/intervention_name") else 'N/A'
    intervention_type = root.findtext(".//intervention/intervention_type") if root.findtext(".//intervention/intervention_type") else 'N/A'
    
    intervention = [intervention_type, intervention_name]
    
    #print(gender, phase, condition, intervention)

    return gender, phase, locations, condition, intervention


def get_str():
    
    str_dict = {}

    # 478504 lines
    with open("../data/trials/all_xml.txt", "r") as file:
        for xml_path in tqdm(file):
            xml_path = f"../data/{xml_path.strip()}"

            # NCT00000150 <- raw_data/NCT0000xxxx/NCT00000150.xml
            nct_id = re.search(r"/([^/]+)\.xml$", xml_path).group(1)
            
            gender, phase, locations, condition, intervention = xmlfile2str(xml_path)
            
            data = {
                "gender": gender,
                "phase": phase,
                "locations": locations,
                "condition": condition,
                "intervention": intervention
            }
                
            str_dict[nct_id] = data
    
    print(len(str_dict))
    
    save_dict(str_dict, '../data/str_dict.pkl')
    
    return str_dict


if __name__ == "__main__":
    
    get_str()

    # age_dict = load_dict('data/age_dict.pkl')
    # print(age_dict['NCT00000102'])
    # print(len(age_dict))