import pandas as pd
import os
from json import dump

SRC_PATH : str  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH : str  = os.path.dirname(SRC_PATH) + '\\csv'
JSON_PATH : str = os.path.dirname(SRC_PATH) + '\\json'


cpp_dump = pd.read_csv(CSV_PATH + '\\TRENNING_PARAMETERS.csv', sep=';', header=None)

param_list = []
for item in cpp_dump[2]:
    param_list.append([item])

for id in range(len(cpp_dump[3])):
    param_list[id].append(cpp_dump[3][id])

for id in range(len(cpp_dump[4])):
    param_list[id].append(cpp_dump[4][id])

for id in range(len(cpp_dump[5])):
    param_list[id].append(cpp_dump[5][id])

for id in range(len(cpp_dump[6])):
    param_list[id].append(cpp_dump[6][id])


user_test_data = {
    cpp_dump[0][i]: {
        'K': cpp_dump[1][i],
        'WEIGHTS': param_list[i]
    } for i in range(358)
}

user_test_data_df = pd.DataFrame(user_test_data)
user_test_data_df.to_json(JSON_PATH + '\\USER_HYPER_PARAMS_HD.json', indent=4)