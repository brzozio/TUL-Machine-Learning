import pandas as pd
import numpy as np
import os
import random
from json import load
from math import sin
from math import cos
from math import sqrt

SRC_PATH : str  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH : str  = os.path.dirname(SRC_PATH) + '\\csv'
JSON_PATH : str = os.path.dirname(SRC_PATH) + '\\json'

NUM_OF_MOVIES = 200
PI = 1.57079632679

with open(JSON_PATH + "\\movie_distance_graph.json", 'r') as file:
    MOVIE_DISTANCE_GRAPH = load(file)
    MOVIE_DISTANCE_GRAPH = {i:{j:MOVIE_DISTANCE_GRAPH[str(i)][str(j)] for j in range(len(MOVIE_DISTANCE_GRAPH[str(i)]))} for i in range(len(MOVIE_DISTANCE_GRAPH))}

with open(JSON_PATH + "\\USER_RATING_DATA.json", 'r') as file:
    user_rating_data = load(file)

# ANGLE_RESOLUTION = 5

# ANGLES = []
# for angle in range(ANGLE_RESOLUTION):
#  ANGLES.append(angle*PI/(ANGLE_RESOLUTION-1))

# SINES = []
# COSINES = []
# for angle in ANGLES:
#  SINES.append(sin(angle))
#  COSINES.append(cos(angle))

# SINES[0] = 0
# SINES[ANGLE_RESOLUTION-1] = 1
# COSINES[0] = 1
# COSINES[ANGLE_RESOLUTION-1] = 0

# PARAMS_WEIGHTS = []

# for i in range(ANGLE_RESOLUTION):
#     for j in range(ANGLE_RESOLUTION):
#         for k in range(ANGLE_RESOLUTION):
#             for l in range(ANGLE_RESOLUTION):
#                 PARAMS_WEIGHTS.append([SINES[i], COSINES[i]*SINES[j], COSINES[i]*COSINES[j]*SINES[k], COSINES[i]*COSINES[j]*COSINES[k]*SINES[l], COSINES[i]*COSINES[j]*COSINES[k]*COSINES[l]])

RESOLUTION = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

PARAMS_WEIGHTS = []

for i in RESOLUTION:
    for j in RESOLUTION:
        for k in RESOLUTION:
            for l in RESOLUTION:
                for m in RESOLUTION:
                    if i+j+k+l+m == 1:
                        PARAMS_WEIGHTS.append([i,j,k,l,m])

def predict(user_id: int, best_weights: list, best_k_neighbours: int) -> float:   


    preditc_ids: list = list(user_rating_data[user]['RATED'].keys())
    train_ids: list = list(user_rating_data[user]['NAN_RATED'].keys())

    weighted_movie_distance = np.zeros((len(MOVIE_DISTANCE_GRAPH),len(MOVIE_DISTANCE_GRAPH)))

    for movie_1 in MOVIE_DISTANCE_GRAPH:
        
        for movie_2 in MOVIE_DISTANCE_GRAPH[movie_1]:                
            
            for id, param in enumerate(MOVIE_DISTANCE_GRAPH[movie_2][movie_1]):
                
                weighted_movie_distance[movie_1][movie_2] += best_weights[id]*param


    training_weighted_movie_distance : list = []

    for movie_1 in weighted_movie_distance:
        
        movie_1 = [element for element in np.argsort(movie_1) if str(element) in train_ids]
        training_weighted_movie_distance.append(movie_1)

    accuracy = 0
    
    for test_movie_id in preditc_ids:
        
        unit_ratings = 0
        for neighbour in range(best_k_neighbours):
            
            # unit_ratings += user_rating_data[user_id]['RATED'][str(training_weighted_movie_distance[int(test_movie_id)][neighbour])]
            unit_ratings += (best_k_neighbours-neighbour-1)*user_rating_data[user_id]['RATED'][str(training_weighted_movie_distance[int(test_movie_id)][neighbour])]
            # print(f"RATING FOR NEIGHTBOUR {neighbour}: {user_rating_data[user_id]['RATED'][str(training_weighted_movie_distance[int(test_movie_id)][neighbour])]}")

        # print(f"User rating: {user_rating_data[user_id]['RATED'][str(test_movie_id)]}, Predicted: {round(unit_ratings/best_k_neighbours)}")

        # if user_rating_data[user_id]['RATED'][str(test_movie_id)] == round(unit_ratings/best_k_neighbours):
        if user_rating_data[user_id]['RATED'][str(test_movie_id)] == round(2*unit_ratings/best_k_neighbours/(best_k_neighbours+1)):
            
            accuracy += 1
            
        
    accuracy = accuracy/len(preditc_ids)

    print(f"ACCURACY: {accuracy}, weights: {best_weights}, K: {best_k_neighbours}")

    return accuracy

user_test_data : list = []

for user in range(10):
        
        #Importowanie z .json
   
        user_test_data.append({
            'USER_ID': user_rating_data[user]['USER_ID'],
            'ACCURACY': accuracy,
            'POPULARITY': best_weights_out[0],
            'RATING': best_weights_out[1],
            'DIRECTOR': best_weights_out[2],
            'ACTORS': best_weights_out[3],
            'GENRES': best_weights_out[4],
            'K': best_k_out,
            'VALIDATION_ID': validation_id
        })

  

user_test_data_df = pd.DataFrame(user_test_data)
user_test_data_df.to_csv(CSV_PATH + '\\USER_DATA_TEST_PREDICTED_HYPER_PYRAMID_WEIGHT_HARMONIC.csv')