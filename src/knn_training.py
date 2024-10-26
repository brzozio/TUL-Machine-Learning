import pandas as pd
import numpy as np
import os
from json import load
from math import sin
from math import cos

BASE_PATH : str = os.path.dirname(os.getcwd())
CSV_PATH  : str = BASE_PATH + '\\csv'
SRC_PATH  : str = BASE_PATH + '\\src'
JSON_PATH : str = BASE_PATH + '\\MACHINE AND DEEP LEARNING\\ML\\json'

NUM_OF_MOVIES = 200
PI = 1.57079632679

with open(JSON_PATH + "\\movie_distance_graph.json", 'r') as file:
    MOVIE_DISTANCE_GRAPH = load(file)
    MOVIE_DISTANCE_GRAPH = {i:{j:MOVIE_DISTANCE_GRAPH[str(i)][str(j)] for j in range(len(MOVIE_DISTANCE_GRAPH[str(i)]))} for i in range(len(MOVIE_DISTANCE_GRAPH))}

with open(JSON_PATH + "\\USER_RATING_DATA.json", 'r') as file:
    user_rating_data = load(file)

ANGLE_RESOLUTION = 5

ANGLES = []
for angle in range(ANGLE_RESOLUTION):
 ANGLES.append(angle*PI/(ANGLE_RESOLUTION-1))

SINES = []
COSINES = []
for angle in ANGLES:
 SINES.append(sin(angle))
 COSINES.append(cos(angle))

SINES[0] = 0
SINES[ANGLE_RESOLUTION-1] = 1
COSINES[0] = 1
COSINES[ANGLE_RESOLUTION-1] = 0

PARAMS_WEIGHTS = []

for i in range(ANGLE_RESOLUTION):
    for j in range(ANGLE_RESOLUTION):
        for k in range(ANGLE_RESOLUTION):
            for l in range(ANGLE_RESOLUTION):
                PARAMS_WEIGHTS.append([SINES[i], COSINES[i]*SINES[j], COSINES[i]*COSINES[j]*SINES[k], COSINES[i]*COSINES[j]*COSINES[k]*SINES[l], COSINES[i]*COSINES[j]*COSINES[k]*COSINES[l]])


def optimize_user(user_id: int, validate_ids: list, training_ids: list, min_k: int = 2, max_k: int = 6) -> tuple[int,int,float]:    

    max_accuracy = 0
    best_weights_id = 0
    best_k_neighbours = 0
    print(f"User id: {user_id}, data: {user_rating_data[user_id]['RATED']}")
    for weights in PARAMS_WEIGHTS:
        
        print(weights[0:1])
        
        weighted_movie_distance = np.zeros((len(MOVIE_DISTANCE_GRAPH),len(MOVIE_DISTANCE_GRAPH)))

        for movie_1 in MOVIE_DISTANCE_GRAPH:
            for movie_2 in MOVIE_DISTANCE_GRAPH[movie_1]:                
                for id, param in enumerate(MOVIE_DISTANCE_GRAPH[movie_2][movie_1]):
                    weighted_movie_distance[movie_1][movie_2] += weights[id]*param

        # conversion of distances to sorted movie ids
        temp_list : list = []

        for movie_1 in weighted_movie_distance:
            movie_1 = [element for element in np.argsort(movie_1) if str(element) in training_ids]
            temp_list.append(movie_1)

        for k in range(min_k, max_k):
            
            accuracy = 0
            for validation_movie_id in validate_ids:
                
                unit_ratings = 0
                for neighbour in range(k):
                    # unit_ratings += user_rating_data[user_id]['RATED']['RATINGS'][int(weighted_movie_distance[validation_movie_id][neighbour])]
                    unit_ratings += user_rating_data[user_id]['RATED'][str(int(temp_list[int(validation_movie_id)][neighbour]))]

                # if user_rating_data[user_id]['RATED']['RATINGS'][validation_movie_id] == round(unit_ratings/k):
                if user_rating_data[user_id]['RATED'][str(validation_movie_id)] == round(unit_ratings/k):
                    accuracy += 1
               
            
            accuracy = accuracy/len(validate_ids)


            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_weights_id = weights
                best_k_neighbours = k
        
                print(f"User: {user_id:<5}, weights: {best_weights_id}, k: {best_k_neighbours}, accuracy: {max_accuracy}")

    return (best_k_neighbours, best_weights_id, max_accuracy)

import random

keys = list(user_rating_data[0]['RATED'].keys())
random.shuffle(keys)

split_index = int(len(keys) * 0.8)
train_keys = keys[:split_index]
test_keys = keys[split_index:]

print(f"TEST: {test_keys}, train: {train_keys}")

validate_ids = test_keys
training_ids = train_keys

print(optimize_user(0, validate_ids, training_ids, 2, 3))