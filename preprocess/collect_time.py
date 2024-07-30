from tqdm import tqdm 
from xml.etree import ElementTree as ET
from datetime import datetime
import re
import pandas as pd
import matplotlib.pyplot as plt
import pickle

"""
    time_dict = { nct_id: [start_date, completion_date, duration] }

"""

def save_dict(file, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(file, f)

def load_dict(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def parse_date(date_str):
    try:
        output = datetime.strptime(date_str, "%B %d, %Y")
    except:
        try:
            output = datetime.strptime(date_str, "%B %Y")
        except Exception as e:
            print(e)
            raise e
    return output

def calculate_duration(start_date, completion_date):
    # Unit: days
    if start_date and completion_date:
        start_date = parse_date(start_date)
        completion_date = parse_date(completion_date)
        duration = (completion_date - start_date).days
    else:
        duration = -1

    return duration

def xmlfile2date(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    try:
        start_date = root.find('start_date').text
    except:
        start_date = ''
    try:
        completion_date = root.find('primary_completion_date').text
    except:
        try:
            completion_date = root.find('completion_date').text 
        except:
            completion_date = ''

    return start_date, completion_date


def get_time():
    
    date_dict = {}

    # 478504 lines
    with open("../data/trials/all_xml.txt", "r") as file:
        for xml_path in tqdm(file):
            xml_path = f"../data/{xml_path.strip()}"

            # NCT00000150 <- raw_data/NCT0000xxxx/NCT00000150.xml
            nct_id = re.search(r"/([^/]+)\.xml$", xml_path).group(1)
            
            start_date, completion_date = xmlfile2date(xml_path)

            #print(start_date, completion_date)

            if start_date and completion_date:
                duration = calculate_duration(start_date, completion_date)
            else:
                duration = -1

            date_dict[nct_id] = [start_date, completion_date, duration]
    
    print(date_dict['NCT00000102'])
    print(len(date_dict))
    
    save_dict(date_dict, '../data/date_dict.pkl')
    
    return date_dict



if __name__ == "__main__":
    get_time()

    # date_dict = load_dict('data/date_dict.pkl')
    # print(date_dict)