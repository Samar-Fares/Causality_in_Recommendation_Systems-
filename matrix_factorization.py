#https://github.com/rajtulluri/Movie-recommendation-system/blob/master/Movie%20recommendation%20System.ipynb

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from surprise.accuracy import mse, mae
from surprise.model_selection import cross_validate
from surprise import Reader, Dataset, SVD 
from collections import defaultdict

def get_data():
    df = pd.read_csv('ratings.csv')
    df.drop('timestamp', axis= 1, inplace= True)
    # print(df.head(5))
    df = df.head(10000)
    df.isna().sum()

    # print("Number of unique movies:", df.movieId.nunique())
    # print("Number of unique users:", df.userId.nunique())

    filter_movies = df.movieId.value_counts() > 1
    filter_movies = filter_movies[filter_movies].index.tolist()

    filter_users = df.userId.value_counts() > 1
    filter_users = filter_users[filter_users].index.tolist()

    # print("Original dimensions:", df.shape)
    df = df[(df.movieId.isin(filter_movies)) & (df.userId.isin(filter_users))]
    # print("New dimensions:", df.shape)

    cols = ['userId', 'movieId', 'rating']
    reader = Reader(rating_scale= (0.5,5))
    data = Dataset.load_from_df(df[cols],reader)

    train_set = data.build_full_trainset()
    antiset = train_set.build_anti_testset()

    mod = SVD(n_epochs= 25, verbose= True)
    cross_validate(mod, data, measures= ['MAE','RMSE'], cv= 5, verbose= True)
    print("Training complete")


    def get_top_n_predictions(predictions,n):
        """
        Return top 'n' predicted ratings for each user
        
        parameters:
            predictions: list of Predictions.Prediction, predictions from the model.
            
            n: int. number of top predictions.
            
        returns:
            top_pred: dict. User and top 'n' predicted ratings for each.
        """
        top_pred = defaultdict(list)
        
        for uid, iid, _, est,_ in pred:
            top_pred[uid].append((iid,est))
            
        for uid, user_rating in top_pred.items():
            user_rating.sort(key= lambda x: x[1], reverse= True)
            top_pred[uid] = user_rating[:n]
            
        return top_pred

    pred = mod.test(antiset)
    top_n = get_top_n_predictions(pred,1)

    def get_user_recommendations(user):
        """
        Returns the recommendations for users specified
        
        parameters:
            user: list. userId
            
        returns:
            recommend: list. List of movieId recommended by model
        """
        recommend = []
        for uid in user:
            recommend.append(top_n[uid])

        return recommend

    user_recommendations = get_user_recommendations(df["userId"])
    df['movieId'] =user_recommendations
    df['movieId'] = df['movieId'].apply(lambda x: x[0][0])

    # print(df)
    # for i in range(5):
    #     print(user_recommendations[i][0][0])
    # for uid, user_ratings in user_recommendations.items():
    #     print(f"User {uid}", user_ratings)

    data = df.to_numpy()

    # print(data.shape)

    cols = ['userId', 'movieId', 'rating']
    reader = Reader(rating_scale= (0.5,5))
    data = Dataset.load_from_df(df[cols],reader)

    train_set = data.build_full_trainset()
    antiset = train_set.build_anti_testset()

    mod = SVD(n_epochs= 25, verbose= True)
    cross_validate(mod, data, measures= ['MAE','RMSE'], cv= 5, verbose= True)
    print("Training complete")

    return(data)
get_data()

