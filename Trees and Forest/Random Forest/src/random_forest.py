import pandas as pd
import os
import numpy as np
import random

SRC_PATH : str  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH : str  = os.path.dirname(SRC_PATH) + '\\csv'
JSON_PATH : str = os.path.dirname(SRC_PATH) + '\\json'


user_rating          = pd.read_json(JSON_PATH + r'\USER_RATING_DATA.json')
USER_HYPER_PARAMS    = pd.read_json(JSON_PATH + r'\USER_HYPER_PARAMS.json')
movie_distance       = pd.read_json(JSON_PATH + r'\movie_distance_graph.json')

def decision_tree(movie_id, all_distances, training_ids, rating_data, threshold, number_of_zero_weights) -> int:

    """
        Dla movie_id wyszukujemyt filmy, które oddalone są od movie_id o mniej niż THRESHOLD.
        Jeśli taki jest, to zapamiętujemy jego ocenę, a potem ze wszystkich wyciątgamy średnią.
    """
    
    decisions: list = []

    for rated_movie in training_ids:

        feature_counter: int = 0

        for feat_dist_i, feature_distance in enumerate(all_distances[movie_id][int(rated_movie)]):

            if feature_distance < threshold[feat_dist_i]:

                feature_counter += 1

        if feature_counter == (5-number_of_zero_weights):

            decisions.append(rating_data[str(rated_movie)])


    if len(decisions) == 0:
        sum = 0
        for rating in rating_data:
            sum += rating_data[rating]
        sum = sum/len(rating_data)
        return np.round(sum)
    
    return int(np.round(np.average(decisions)))

def test_user(user, best_weights, predict_ids, train_ids, num_of_trees) -> tuple[list, list]:

    from collections import Counter

    def manipulate_weights(weights, min_percent=0.95, max_percent=1.05) -> tuple:
        # import random
      
        # zero_index = random.randint(0, len(weights) - 1)
        
        # new_weights = [
        #     0 if i == zero_index else round(w * random.uniform(min_percent, max_percent), 2)
        #     for i, w in enumerate(weights)
        # ]
        
        # return new_weights

        import random

        num_zero_weights = random.randint(0, min(4, len(weights)))
        zero_indices     = random.sample(range(len(weights)), num_zero_weights)

        new_weights = [
            0 if i in zero_indices else round(w * random.uniform(min_percent, max_percent), 2)
            for i, w in enumerate(weights)
        ]

        return (new_weights, num_zero_weights)

    def select_random_subset(training_ids_inner, percentage):
       
        import random

        num_items_to_select = int(len(training_ids_inner) * percentage)
        
        selected_subset = random.sample(training_ids_inner, num_items_to_select)
        
        return selected_subset
    
    ratings: list = []
    movies:  list = []

    for predict_movie in predict_ids:

        temp_rating_list: list = []

        for _ in range(num_of_trees):

            weights_tree, num_zero_weights = manipulate_weights(best_weights)

            training_ids_tree = select_random_subset(training_ids_inner=train_ids, percentage=0.9)

            output = decision_tree(movie_id=int(predict_movie), all_distances=movie_distance, training_ids=training_ids_tree, rating_data=user_rating["RATED"][user], threshold=weights_tree, number_of_zero_weights=num_zero_weights)
            temp_rating_list.append(output)

        count = Counter(temp_rating_list)
        movie_rating = count.most_common(1)[0][0]

        ratings.append(movie_rating)
        movies.append(predict_movie)

    return (movies, ratings)


from json import dump, load
user_rating_data_predicted : list = []
user_rating_data_movies: list = []

NUM_OF_USERS : int = len(user_rating)

for user in range(NUM_OF_USERS):
    
    user_id :    int  = user_rating['USER_ID'][user]
    best_weight: list = USER_HYPER_PARAMS[user_id]
    preditc_ids: list = list(user_rating['NAN_RATED'][user].keys())
    train_ids:   list = list(user_rating['RATED'][user].keys())

    returned_movies, returned_reviews = test_user(user=user, best_weights=best_weight, predict_ids=preditc_ids, train_ids=train_ids, num_of_trees=200)
    
    user_rating_data_movies.append(returned_movies)
    user_rating_data_predicted.append(returned_reviews)

    print(f"User {user} predicting data ...")


user_test_data = {
    user_rating['USER_ID'][i]: {
       'MOVIES' : [str(int(movie_id) + 1) for movie_id in user_rating_data_movies[i]],
       'RATINGS' : user_rating_data_predicted[i]
    }
    for i in range(NUM_OF_USERS)
}

user_test_data_df = pd.DataFrame(user_test_data)
user_test_data_df.to_json(JSON_PATH + '\\USER_PREDICTED_RATINGS.json', indent=4)

with open(JSON_PATH + '\\USER_PREDICTED_RATINGS.json', 'r') as f:
    data = load(f)

user_test_data = {}
for user_id, ratings_data in data.items():
    movies = ratings_data['MOVIES']
    ratings = ratings_data['RATINGS']
    
    user_test_data[user_id] = {movie: rating for movie, rating in zip(movies, ratings)}


with open(JSON_PATH + '\\USER_PREDICTED_RATINGS_TRANSFORMED.json', 'w') as f:
    dump(user_test_data, f, indent=4)