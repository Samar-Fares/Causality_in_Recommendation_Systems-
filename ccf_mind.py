from itertools import count
import pandas as pd

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import SVD
from surprise import accuracy
from causalnex.network import BayesianNetwork
from causalnex.structure.notears import from_pandas
import random
from collections import defaultdict



# from surprise.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from causalnex.inference import InferenceEngine

def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

def traditional():

    behaviors_colnames=['id', 'user_id', 'time', 'history', 'impressions'] 
    behaviors_df = pd.read_csv('MINDlarge_dev/behaviors.tsv', sep='\t', names=behaviors_colnames, header=None)
    behaviors_df = behaviors_df.head(1000)
    counter = 0
    df = pd.DataFrame(columns=['user_id','new_id','rating', 'history'])
    users = []
    news = []
    ratings = []
    history = []
    for i in range(1000):
        # print(f'i {i}')
        # print(len(behaviors_df['impressions'][i].split(" ")))
        # for j in range(len(behaviors_df['impressions'][i].split(" "))):
            # if(behaviors_df['impressions'][i].split(' ')[j].split('-')[1].split(" ")[0] == '1'):
                users.append(float(behaviors_df['user_id'][i].replace("U",'')))
                if(is_float(behaviors_df['history'][i])):
                    history.append(0)
                else:
                    history.append(float(behaviors_df['history'][i].split(' ')[0].replace("N",'')))
                news.append(float(behaviors_df['impressions'][i].split(' ')[0].split('-')[0].replace("N", '')))
                ratings.append(float(behaviors_df['impressions'][i].split(' ')[0].split('-')[1].split(" ")[0]))
                counter = counter + 1

    df['user_id'] = users
    df['new_id'] = news
    df['rating']  = ratings
    df['history']  = history


    behaviors_colnames1=['id', 'user_id', 'time', 'history', 'impressions'] 
    behaviors_df1 = pd.read_csv('MINDlarge_test/behaviors.tsv', sep='\t', names=behaviors_colnames1, header=None)
    behaviors_df1 = behaviors_df.head(1000)
    counter = 0
    df1 = pd.DataFrame(columns=['user_id','new_id','history'])
    users1 = []
    news1 = []
    history1 = []
    for i in range(1000):
        # print(f'i {i}')
        # print(len(behaviors_df['impressions'][i].split(" ")))
        # for j in range(len(behaviors_df1['impressions'][i].split(" "))):
            # if(behaviors_df['impressions'][i].split(' ')[j].split('-')[1].split(" ")[0] == '1'):
                users1.append(float(behaviors_df1['user_id'][i].replace("U",'')))
                if(is_float(behaviors_df1['history'][i])):
                    history1.append(0)
                else:
                    history1.append(float(behaviors_df1['history'][i].split(' ')[0].replace("N",'')))   
                if(is_float(behaviors_df1['impressions'][i])):
                    news1.append(0)
                else:
                    news1.append(float(behaviors_df1['impressions'][i].split(' ')[0].split('-')[0].replace("N",'')))             
                # news1.append(float(behaviors_df1['impressions'][i].split(' ')[0].replace("N", '')))
                counter = counter + 1

    df1['user_id'] = users1
    df1['new_id'] = news1
    df1['history']  = history1

    return df, df1

# behaviors_colnames=['id', 'user_id', 'time', 'history', 'impressions'] 
# behaviors_df = pd.read_csv('MINDlarge_train/behaviors.tsv', sep='\t', names=behaviors_colnames, header=None)
# # behaviors_df = behaviors_df.head(20)
# behaviors_df['user_id'] = behaviors_df['user_id'].apply(lambda x: x.replace("U",''))
# behaviors_df['new_id'] = behaviors_df['impressions'].apply(lambda x: x.split('-')[0].replace("N", ''))
# behaviors_df['rating'] = behaviors_df['impressions'].apply(lambda x: x.split('-')[1].split(" ")[0])


# df = pd.DataFrame()
# df['user_id'] = behaviors_df['user_id'].values
# df['new_id'] = behaviors_df['new_id'].values
# df['rating']  = behaviors_df['rating'].values
# Creation of the dataframe. Column names are irrelevant.
# ratings_dict = {'itemID': [1, 1, 1, 2, 2],
#                 'userID': [9, 32, 2, 45, 'user_foo'],
#                 'rating': [3, 2, 4, 3, 1]}
# df = pd.DataFrame(ratings_dict)




def rec_intervention(data):

    sm = from_pandas(data)
        # Removing the learned edge from the model
    sm.remove_edge("user_id", "new_id")
    # Changing the direction of the learned edge
    sm.remove_edge("new_id", "user_id")
    sm.remove_edge("rating", "user_id")
    sm.remove_edge("rating", 'new_id')

    

    # sm.add_edge("d", "c", origin="learned")
    # # Adding the edge that was not learned by the algorithm
    # sm.add_edge("a", "e", origin="expert")
    bn = BayesianNetwork(sm)
    bn.fit_node_states(data)

    print(data.shape[1])
    bn = bn.fit_cpds(data, method="BayesianEstimator", bayes_prior="K2")

    ie = InferenceEngine(bn)    
    # Doing an intervention to the node "d"
    ie.do_intervention("new_id", random.choise(data['new_id']))
    # Querying all the updated marginal probabilities of the model's distribution
    marginals_after_interventions = ie.query({})
    # Re-introducing the original conditional dependencies
    ie.reset_do("d")
    # return df 

def similarity(X):
        # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 5))

    # The columns must correspond to user id, item id and ratings (in that order).
    # data = Dataset.load_from_df(df[['user_id', 'new_id', 'rating']], reader)
    # trainset, testset = train_test_split(data, test_size=.25)
    # trainset = df.iloc[:, :200]
    # testset = df.iloc[:, 200:]
    # X = df.pivot(index = 'user_id', columns = 'new_id', values = 'rating').fillna(0)
    # Y = testset.pivot(index = 'userId', columns = 'new_id', values = 'rating').fillna(0)
    # data = Dataset.load_from_df(X[['user_id', 'new_id', 'rating']], reader)
    # trainset, testset = train_test_split(X, test_size=.25)

    # # data = Dataset.load_from_df(Y[['user_id', 'new_id', 'rating']], reader)




    # # # We can now use this dataset as we please, e.g. calling cross_validate


    # # We'll use the famous SVD algorithm.
    # algo = SVD()
    # # cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # # Train the algorithm on the trainset, and predict ratings for the testset
    # algo.fit(trainset)
    # predictions = algo.test(testset)

    # # Then compute RMSE
    # accuracy.rmse(predictions)
    X_train, X_test = train_test_split(X, test_size = 0.30, random_state = 42)

    user_data = X_train.pivot(index = 'user_id', columns = 'new_id', values = 'rating').fillna(0)

    # make a copy of train and test datasets
    dummy_train = X_train.copy()
    dummy_test = X_test.copy()

    # dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x > 0 else 1)
    # dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x > 0 else 0)

    # The movies not rated by user is marked as 1 for prediction 
    dummy_train = dummy_train.pivot(index = 'user_id', columns = 'new_id', values = 'rating').fillna(1)

    # The movies not rated by user is marked as 0 for evaluation 
    dummy_test = dummy_test.pivot(index ='user_id', columns = 'new_id', values = 'rating').fillna(0)


    # User Similarity Matrix using Cosine similarity as a similarity measure between Users
    user_similarity = cosine_similarity(user_data)
    user_similarity[np.isnan(user_similarity)] = 0
    # print(user_similarity)
    # print(user_similarity.shape)


    user_predicted_ratings = np.dot(user_similarity, user_data)
    user_predicted_ratings


    # np.multiply for cell-by-cell multiplication 

    user_final_ratings = np.multiply(user_predicted_ratings, dummy_train)
    user_final_ratings.head()


    ##############Evaluation#################

    test_user_features = X_test.pivot(index = 'user_id', columns = 'new_id', values = 'rating').fillna(0)
    test_user_similarity = cosine_similarity(test_user_features)
    test_user_similarity[np.isnan(test_user_similarity)] = 0

    # print(test_user_similarity)
    # print("- "*10)
    # print(test_user_similarity.shape)

    user_predicted_ratings_test = np.dot(test_user_similarity, test_user_features)
    user_predicted_ratings_test

    test_user_final_rating = np.multiply(user_predicted_ratings_test, dummy_test)
    test_user_final_rating.head()



    X = test_user_final_rating.copy() 
    X = X[X > 0] # only consider non-zero values as 0 means the user haven't rated the movies

    scaler = MinMaxScaler(feature_range = (0.5, 5))
    scaler.fit(X)
    pred = scaler.transform(X)

    # print(pred)

    # total non-NaN value
    total_non_nan = np.count_nonzero(~np.isnan(pred))
    total_non_nan

    test = X_test.pivot(index = 'user_id', columns = 'new_id', values = 'rating')
    test.head()

    # RMSE Score

    diff_sqr_matrix = (test - pred)**2
    sum_of_squares_err = diff_sqr_matrix.sum().sum() # df.sum().sum() by default ignores null values

    rmse = np.sqrt(sum_of_squares_err/total_non_nan)
    print(rmse)

    # Mean abslute error

    mae = np.abs(pred - test).sum().sum()/total_non_nan
    print(mae)

def svd(df):
    cols = ['user_id','rating', 'history']
    reader = Reader(rating_scale= (0,1))
    data = Dataset.load_from_df(df[cols],reader)

    train_set = data.build_full_trainset()
    antiset = train_set.build_anti_testset()

    mod = SVD(n_epochs= 25, verbose= True)
    cross_validate(mod, data, measures= ['MAE','RMSE'], cv= 5, verbose= True)
    print("Training complete")
    pred = mod.test(antiset)
    return pred

df = traditional()[0]
interv = traditional()[1]
svd(df)
# pred = svd(df)
# rec_intervention(traditional()[0])

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


# top_n = get_top_n_predictions(pred,1)

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

# user_recommendations = get_user_recommendations(df["user_id"])
users = []
for i in range(len(df)):
    # print(df['user_id'][i] not in users)
    # if(df['user_id'][i] not in users):
        df['history'][i] = interv.loc[(interv['user_id'] ==  df['user_id' ][i]) , 'history'].iloc[0]
        users.append(df['user_id' ][i])
# for i in range(len(df)):
  
    #  if(df['user_id'][i] not in users):
    #      print("in if")
    #      df = df.drop(labels=i, axis=0)
# df['new_id'] = df['new_id'].apply(lambda x: x[0][0])
svd(df)
# for i in range(5):
#     print(user_recommendations[i][0][0])
# for uid, user_ratings in user_recommendations.items():
#     print(f"User {uid}", user_ratings)
