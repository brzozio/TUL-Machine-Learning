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
    USER_RATING_DATA = load(file)

with open(JSON_PATH + "\\USER_HYPER_PARAMS.json", 'r') as file:
    USER_HYPER_PARAMS = load(file)

RESOLUTION = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

PARAMS_WEIGHTS = []

for i in RESOLUTION:
    for j in RESOLUTION:
        for k in RESOLUTION:
            for l in RESOLUTION:
                for m in RESOLUTION:
                    if i+j+k+l+m == 1:
                        PARAMS_WEIGHTS.append([i,j,k,l,m])

def predict(user_id: int, user_iter: int, best_weights: list, best_k_neighbours: int, predict_ids: list, train_ids: list) -> tuple[list, list]:   

    returned_movies: list = []
    returned_reviews: list = []

    weighted_movie_distance = np.zeros((len(MOVIE_DISTANCE_GRAPH),len(MOVIE_DISTANCE_GRAPH)))

    for movie_1 in MOVIE_DISTANCE_GRAPH:
        
        for movie_2 in MOVIE_DISTANCE_GRAPH[movie_1]:                
            
            for id, param in enumerate(MOVIE_DISTANCE_GRAPH[movie_2][movie_1]):
                
                weighted_movie_distance[movie_1][movie_2] += best_weights[id]*param


    training_weighted_movie_distance : list = []

    for movie_1 in weighted_movie_distance:
        
        movie_1 = [element for element in np.argsort(movie_1) if str(element) in train_ids]
        training_weighted_movie_distance.append(movie_1)

    # accuracy = 0
    
    for test_movie_id in predict_ids:
        
        unit_ratings = 0
        for neighbour in range(best_k_neighbours):
            
            unit_ratings += USER_RATING_DATA[user_iter]['RATED'][str(training_weighted_movie_distance[int(test_movie_id)][neighbour])]
            # unit_ratings += (best_k_neighbours-neighbour-1)*USER_RATING_DATA[user_iter]['RATED'][str(training_weighted_movie_distance[int(test_movie_id)][neighbour])]
            # print(f"RATING FOR NEIGHTBOUR {neighbour}: {user_rating_data[user_id]['RATED'][str(training_weighted_movie_distance[int(test_movie_id)][neighbour])]}")
      
        # if user_rating_data[user_id]['RATED'][str(test_movie_id)] == round(unit_ratings/best_k_neighbours):
        # if USER_RATING_DATA[user_iter]['RATED'][str(test_movie_id)] == round(2*unit_ratings/best_k_neighbours/(best_k_neighbours+1)):
            
        #     accuracy += 1

        rating = round(round(unit_ratings/best_k_neighbours))
        
        returned_movies.append(test_movie_id)
        returned_reviews.append(rating)

    # accuracy = accuracy/len(preditc_ids)

    # print(f"ACCURACY: {accuracy}, weights: {best_weights}, K: {best_k_neighbours}")

    return (returned_movies, returned_reviews)

user_rating_data_predicted : list = []
user_rating_data_movies: list = []

NUM_OF_USERS : int = len(USER_RATING_DATA)

for user in range(NUM_OF_USERS):
    
    user_id :    int  = USER_RATING_DATA[user]['USER_ID']
    best_weight: list = USER_HYPER_PARAMS[str(user_id)]['WEIGHTS']
    best_k      : int  = USER_HYPER_PARAMS[str(user_id)]['K']
    preditc_ids: list = list(USER_RATING_DATA[user]['NAN_RATED'].keys())
    train_ids:   list = list(USER_RATING_DATA[user]['RATED'].keys())

    # print(f"{user:<3} User: {user_id:<4} Weights: {best_weight} K: {best_k}")
        
    returned_movies, returned_reviews = predict(user_id=user_id, user_iter=user, best_weights=best_weight, best_k_neighbours=best_k, predict_ids=preditc_ids, train_ids=train_ids)
    
    user_rating_data_movies.append(returned_movies)
    user_rating_data_predicted.append(returned_reviews)
    

for i in range(3):
    print(f"User {USER_RATING_DATA[i]['USER_ID']} - Movies: {user_rating_data_movies[i]}, Reviews: {user_rating_data_predicted[i]}")


user_test_data = {
    USER_RATING_DATA[i]['USER_ID']: {
       'MOVIES' : user_rating_data_movies[i],
       'RATINGS' : user_rating_data_predicted[i]
    }
    for i in range(NUM_OF_USERS)
}


user_test_data_df = pd.DataFrame(user_test_data)
user_test_data_df.to_json(JSON_PATH + '\\USER_PREDICTED_RATINGS.json', indent=4)