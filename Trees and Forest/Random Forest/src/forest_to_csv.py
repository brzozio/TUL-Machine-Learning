import pandas as pd
import os

SRC_PATH : str  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH : str  = os.path.dirname(SRC_PATH) + '\\csv'
JSON_PATH : str = os.path.dirname(SRC_PATH) + '\\json'

df = pd.read_csv(CSV_PATH +'\\task.csv', delimiter=';', names=['ID', 'USER_ID', 'MOVIE_ID', 'RATING'])
user_movie_dict = df.groupby('USER_ID')['MOVIE_ID'].apply(list).to_dict()

raw_null_csv = pd.read_csv(CSV_PATH + '\\task.csv', sep=';', header=None).fillna(-1)
raw_null_csv.columns = ['ID', 'USER_ID', 'MOVIE_ID', 'RATING']

import json
with open(JSON_PATH + '\\USER_PREDICTED_RATINGS_TRANSFORMED.json', 'r') as f:
    data = json.load(f)

raw_null_csv['RATING'] = raw_null_csv['RATING'].astype(int)

for index, row in raw_null_csv.iterrows():
    user_id = row['USER_ID']
    movie_id = row['MOVIE_ID']

    user_rating = data[str(int(user_id))][str(int(movie_id))]
    print(f"Rating for user: {user_id} and movie: {movie_id} is: {user_rating}")

    raw_null_csv.at[index, 'RATING'] = int(user_rating)  

raw_null_csv.to_csv(CSV_PATH + '\\submission_forest.csv', header=None, index=None, sep=';')