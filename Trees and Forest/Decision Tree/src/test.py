
import pandas as pd
import os
import numpy as np
import random

SRC_PATH : str  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH : str  = os.path.dirname(SRC_PATH) + '\\csv'
JSON_PATH : str = os.path.dirname(SRC_PATH) + '\\json'

node1 = 0
node2 = 0
node3 = 0
node4 = 0
node5 = 0


user_rating = pd.read_json(JSON_PATH + r'\USER_RATING_DATA.json')
USER_HYPER_PARAMS = pd.read_json(JSON_PATH + r'\USER_HYPER_PARAMS.json')
movie_distance = pd.read_json(JSON_PATH + r'\movie_distance_graph.json')

def decision_tree(movie_id, all_distances, training_ids, rating_data, threshold) -> int:

    """
        Dla movie_id wyszukujemyt filmy, które oddalone są od movie_id o mniej niż THRESHOLD.
        Jeśli taki jest, to zapamiętujemy jego ocenę, a potem ze wszystkich wyciątgamy średnią.
    """
    
    decisions: list = []
    global node1, node2, node3, node4, node5

    for rated_movie in training_ids:

        feat_dist_i = 0

        if all_distances[movie_id][int(rated_movie)][0] < threshold[0]:
            node1 += 1
            if all_distances[movie_id][int(rated_movie)][1] < threshold[1]:
                node2 += 1
                if all_distances[movie_id][int(rated_movie)][2] < threshold[2]:
                    node3 += 1
                    if all_distances[movie_id][int(rated_movie)][3] < threshold[3]:
                        node4 += 1
                        if all_distances[movie_id][int(rated_movie)][4] < threshold[4]:
                            node5 += 1
                            decisions.append(rating_data[str(rated_movie)])

        # if feat_dist_i == 4:
        #     decisions.append(rating_data[str(rated_movie)])
        #     node5 += 1
        # elif feat_dist_i == 3: 
        #     node4 += 1
        # elif feat_dist_i == 2: 
        #     node3 += 1
        # elif feat_dist_i == 1: 
        #     node2 += 1
        # elif feat_dist_i == 0: 
        #     node1 += 1

    if len(decisions) == 0:
        sum = 0
        for rating in rating_data:
            sum += rating_data[rating]
        sum = sum/len(rating_data)
        return int(np.round(sum))
    
    return int(np.round(np.average(decisions)))

def test_user(user, best_weights, predict_ids, train_ids) -> tuple[list, list]:

    predict_movie = predict_ids[0]

    output = decision_tree(movie_id=int(predict_movie), all_distances=movie_distance, training_ids=train_ids, rating_data=user_rating["RATED"][user], threshold=best_weights)

    return (None, None)

user_id :    int  = user_rating['USER_ID'][9]
best_weight: list = USER_HYPER_PARAMS[user_id]
preditc_ids: list = list(user_rating['NAN_RATED'][9].keys())
train_ids:   list = list(user_rating['RATED'][9].keys())

returned_movies, returned_reviews = test_user(user=9, best_weights=best_weight, predict_ids=preditc_ids, train_ids=train_ids)


print(node1)
print(node2)
print(node3)
print(node4)
print(node5)