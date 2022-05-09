import pandas as pd
import numpy as np

def get_data():

    cols = ['userId', 'movieId', 'rating', 'time']

    df = pd.read_csv('ratings.csv', sep=',', names=cols, header=None)
    df = df.head(2000)
    df.isna().sum()

    # print("Number of unique movies:", df.movieId.nunique())
    # print("Number of unique users:", df.userId.nunique())

    filter_movies = df.movieId.value_counts() > 3
    filter_movies = filter_movies[filter_movies].index.tolist()

    filter_users = df.userId.value_counts() > 3
    filter_users = filter_users[filter_users].index.tolist()

    # print("Original dimensions:", df.shape)
    df = df[(df.movieId.isin(filter_movies)) & (df.userId.isin(filter_users))]


    data = np.vstack([df['userId'],df['movieId'], df['rating']])

    data = np.transpose(data)

    return data





