from tqdm import tqdm 
from xml.etree import ElementTree as ET
from datetime import datetime
import re
import pandas as pd
import pickle

"""
    location_dict = {
        nct_id: {"countries": countries, 
                "states": states, 
                "cities": cities
                }
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
    
    locations_info = []
    
    countries = []
    states = []
    cities = []
    
    for facility in root.findall(".//location/facility"):
        country = facility.findtext(".//country")
        city = facility.findtext(".//city")
        state = facility.findtext(".//state")
        if country:
            countries.append(country)
        if city:
            cities.append(city)
        if state:
            states.append(state)
            
    countries = list(set(countries))
    states = list(set(states))
    cities = list(set(cities))

    return countries, states, cities


def get_location():
    
    
    location_dict = {}

    # 478504 lines
    with open("../data/trials/all_xml.txt", "r") as file:
        for xml_path in tqdm(file):
            xml_path = f"../data/{xml_path.strip()}"

            # NCT00000150 <- raw_data/NCT0000xxxx/NCT00000150.xml
            nct_id = re.search(r"/([^/]+)\.xml$", xml_path).group(1)
            
            countries, states, cities = xmlfile2str(xml_path)

                
            location_dict[nct_id] = {"countries": countries, "states": states, "cities": cities}
    
    print(len(location_dict))
    
    save_dict(location_dict, '../data/location_dict.pkl')
    
    return location_dict


if __name__ == "__main__":
    
    get_location()
    
    # location_dict = load_dict('data/location_dict.pkl')
    # ctr = 0
    # for nct_id, location in location_dict.items():
    #     print(nct_id)
    #     print(location)
    #     if ctr == 20:
    #         break
    #     ctr += 1

    # location_dict = load_dict('data/location_dict.pkl')
    # print(location_dict['NCT00000102'])
    # print(len(location_dict))