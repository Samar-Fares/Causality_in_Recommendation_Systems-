
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import SVD


# read ratings file
ratings = pd.read_csv('ratings.csv')


# X_train, X_test = train_test_split(ratings, test_size = 0.30, random_state = 42)

# # pivot ratings into movie features
# user_data = X_train.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
# # print(user_data.head())

# # make a copy of train and test datasets
# dummy_train = X_train.copy()
# dummy_test = X_test.copy()

# dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x > 0 else 1)
# dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x > 0 else 0)

# # The movies not rated by user is marked as 1 for prediction 
# dummy_train = dummy_train.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(1)

# # The movies not rated by user is marked as 0 for evaluation 
# dummy_test = dummy_test.pivot(index ='userId', columns = 'movieId', values = 'rating').fillna(0)


# # User Similarity Matrix using Cosine similarity as a similarity measure between Users
# user_similarity = cosine_similarity(user_data)
# user_similarity[np.isnan(user_similarity)] = 0
# # print(user_similarity)
# # print(user_similarity.shape)


# user_predicted_ratings = np.dot(user_similarity, user_data)
# user_predicted_ratings


# # np.multiply for cell-by-cell multiplication 

# user_final_ratings = np.multiply(user_predicted_ratings, dummy_train)
# user_final_ratings.head()


# ##############Evaluation#################

# test_user_features = X_test.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)
# test_user_similarity = cosine_similarity(test_user_features)
# test_user_similarity[np.isnan(test_user_similarity)] = 0

# # print(test_user_similarity)
# # print("- "*10)
# # print(test_user_similarity.shape)

# user_predicted_ratings_test = np.dot(test_user_similarity, test_user_features)
# user_predicted_ratings_test

# test_user_final_rating = np.multiply(user_predicted_ratings_test, dummy_test)
# test_user_final_rating.head()

# ratings['rating'].describe()


# X = test_user_final_rating.copy() 
# X = X[X > 0] # only consider non-zero values as 0 means the user haven't rated the movies

# scaler = MinMaxScaler(feature_range = (0.5, 5))
# scaler.fit(X)
# pred = scaler.transform(X)

# # print(pred)

# # total non-NaN value
# total_non_nan = np.count_nonzero(~np.isnan(pred))
# total_non_nan

# test = X_test.pivot(index = 'userId', columns = 'movieId', values = 'rating')
# test.head()

# # RMSE Score

# diff_sqr_matrix = (test - pred)**2
# sum_of_squares_err = diff_sqr_matrix.sum().sum() # df.sum().sum() by default ignores null values

# rmse = np.sqrt(sum_of_squares_err/total_non_nan)
# print(rmse)

# # Mean abslute error

# mae = np.abs(pred - test).sum().sum()/total_non_nan
# print(mae)


def svd(df):
    cols = ['userId','movieId', 'rating']
    reader = Reader(rating_scale= (0,1))
    data = Dataset.load_from_df(df[cols],reader)

    train_set = data.build_full_trainset()
    antiset = train_set.build_anti_testset()

    mod = SVD(n_epochs= 25, verbose= True)
    cross_validate(mod, data, measures= ['MAE','RMSE'], cv= 5, verbose= True)
    print("Training complete")
    pred = mod.test(antiset)
    return pred


svd(ratings)