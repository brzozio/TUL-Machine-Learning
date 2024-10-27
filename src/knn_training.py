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


def optimize_user(user_id: int, validate_ids: list, training_ids: list, min_k: int = 2, max_k: int = 6) -> tuple[int,list,float]:    

    max_accuracy = 0
    best_weights_id = [0,0,0,0,0]
    best_k_neighbours = min_k
    # print(f"User id: {user_id}, data: {user_rating_data[user_id]['RATED']}")
    for weights in PARAMS_WEIGHTS:
        
        # print(weights[0:1])
        
        weighted_movie_distance = np.zeros((len(MOVIE_DISTANCE_GRAPH),len(MOVIE_DISTANCE_GRAPH)))

        for movie_1 in MOVIE_DISTANCE_GRAPH:
            for movie_2 in MOVIE_DISTANCE_GRAPH[movie_1]:                
                for id, param in enumerate(MOVIE_DISTANCE_GRAPH[movie_2][movie_1]):
                    weighted_movie_distance[movie_1][movie_2] += weights[id]*param

        # conversion of distances to sorted movie ids
        training_ids_sorted : list = []

        for movie_1 in weighted_movie_distance:
            movie_1 = [element for element in np.argsort(movie_1) if str(element) in training_ids]
            training_ids_sorted.append(movie_1)

        for k in range(min_k, max_k):
            
            accuracy = 0
            for validation_movie_id in validate_ids:
                
                unit_ratings = 0
                for neighbour in range(k):
                    # unit_ratings += user_rating_data[user_id]['RATED']['RATINGS'][int(weighted_movie_distance[validation_movie_id][neighbour])]
                    
                    # working: mean of k neighbours
                    # unit_ratings += user_rating_data[user_id]['RATED'][str(training_ids_sorted[int(validation_movie_id)][neighbour])]

                    # weighted mean o k neighbours via harmonic descent
                    unit_ratings += (k-neighbour-1)*user_rating_data[user_id]['RATED'][str(training_ids_sorted[int(validation_movie_id)][neighbour])]

                # if user_rating_data[user_id]['RATED']['RATINGS'][validation_movie_id] == round(unit_ratings/k):
                
                #working: mean of k neighbours
                #if user_rating_data[user_id]['RATED'][str(validation_movie_id)] == round(unit_ratings/k):

                if user_rating_data[user_id]['RATED'][str(validation_movie_id)] == round(2*unit_ratings/k/(k+1)):
                    accuracy += 1
               
            
            accuracy = accuracy/len(validate_ids)


            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_weights_id = weights
                best_k_neighbours = k
        
                # print(f"User: {user_id:<5}, weights: {best_weights_id}, k: {best_k_neighbours}, accuracy: {max_accuracy}")

    return (best_k_neighbours, best_weights_id, max_accuracy)

def test_user(user_id: int, best_weights: list, test_ids: list, train_ids: list, best_k_neighbours: int) -> float:    

  
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
    
    for test_movie_id in test_ids:
        
        unit_ratings = 0
        for neighbour in range(best_k_neighbours):
            
            # unit_ratings += user_rating_data[user_id]['RATED'][str(training_weighted_movie_distance[int(test_movie_id)][neighbour])]
            unit_ratings += (best_k_neighbours-neighbour-1)*user_rating_data[user_id]['RATED'][str(training_weighted_movie_distance[int(test_movie_id)][neighbour])]
            # print(f"RATING FOR NEIGHTBOUR {neighbour}: {user_rating_data[user_id]['RATED'][str(training_weighted_movie_distance[int(test_movie_id)][neighbour])]}")

        # print(f"User rating: {user_rating_data[user_id]['RATED'][str(test_movie_id)]}, Predicted: {round(unit_ratings/best_k_neighbours)}")

        # if user_rating_data[user_id]['RATED'][str(test_movie_id)] == round(unit_ratings/best_k_neighbours):
        if user_rating_data[user_id]['RATED'][str(test_movie_id)] == round(2*unit_ratings/best_k_neighbours/(best_k_neighbours+1)):
            
            accuracy += 1
            
        
    accuracy = accuracy/len(test_ids)

    print(f"ACCURACY: {accuracy}, weights: {best_weights}, K: {best_k_neighbours}")

    return accuracy

user_test_data : list = []


NUM_OF_CROSS_VALIDATION = 5

for user in range(10):
    print(user)
    for validation_id in range(NUM_OF_CROSS_VALIDATION):

        keys = list(user_rating_data[user]['RATED'].keys())
        random.shuffle(keys)
        split_index = int(len(keys) * 0.9)
        train_valid_keys = keys[:split_index]
        test_keys = keys[split_index:]

        keys = list(train_valid_keys)
        random.shuffle(keys)
        split_index = int(len(keys) * 0.85)
        train_keys = keys[:split_index]
        valid_keys = keys[split_index:]


        test_ids = test_keys
        validate_ids = valid_keys
        training_ids = train_keys

        # print(f"Train keys: {train_keys}, \nvalid keys: {valid_keys}, \ntest keys: {test_keys}")

        best_k_out, best_weights_out, _ = optimize_user(user, validate_ids, training_ids, 2, 7)

        accuracy = test_user(user_id=user, best_weights=best_weights_out, best_k_neighbours=best_k_out, test_ids=test_ids, train_ids=train_valid_keys)

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