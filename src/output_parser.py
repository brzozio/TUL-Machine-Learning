import numpy as np
import os
from json import load

SRC_PATH : str  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH : str  = os.path.dirname(SRC_PATH) + '\\csv'
JSON_PATH : str = os.path.dirname(SRC_PATH) + '\\json'

def load_user_predicted_ratings():

    with open(JSON_PATH + "\\USER_PREDICTED_RATINGS.json", 'r') as file:
        constructor = load(file)

    data = {int(u_id):{
                    int(constructor[u_id]['MOVIES'][item_id]):int(constructor[u_id]['RATINGS'][item_id]) for item_id in range(len(constructor[u_id]['RATINGS']))
                    } for u_id in constructor}
    return data

# USER_PREDICTED_RATINGS[user_id][movie_id] = movie_rating
USER_PREDICTED_RATINGS = load_user_predicted_ratings()

submission_data = np.genfromtxt(CSV_PATH + "\\task.csv", delimiter=';')
total_misses = 0

for prediction in submission_data:
    # movie id generated in USER_PREDICTED_RATINGS is 0 indexed vs 1 indexed in movie _id in task.csv
    if int(prediction[2])-1 in USER_PREDICTED_RATINGS[int(prediction[1])]:
        prediction[3] = int(USER_PREDICTED_RATINGS[int(prediction[1])][int(prediction[2])-1])
    else:
        print(f'no movie_id: {int(prediction[2])-1} in user_id: {int(prediction[1])}')

print(submission_data)
np.savetxt(X=submission_data, fname=CSV_PATH + "\\submission.csv", delimiter=";")

