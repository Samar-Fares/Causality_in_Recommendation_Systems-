# import code
# code.interact(local=locals)
from itertools import count
import re
from tkinter.messagebox import NO
import pandas as pd
import numpy as np
import math
from causalnex.structure.notears import from_pandas
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from IPython.display import Image
from scipy.stats import chi2_contingency


# function optimized to run on gpu 
def get_data():
    behaviors_colnames=['id', 'user_id', 'time', 'history', 'impressions'] 
    news_colnames=['id', 'category', 'sub', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
    embedding_colnames=['id', 'vector']

    behaviors_df = pd.read_csv('MINDlarge_train/behaviors.tsv', sep='\t', names=behaviors_colnames, header=None)
    behaviors_df = behaviors_df.head(50)
    news_df = pd.read_csv('MINDlarge_train/news.tsv', sep='\t', names=news_colnames, header=None)
    entity_embeddings_df = pd.read_csv('MINDlarge_train/entity_embedding.vec', sep='\t', names=embedding_colnames, header=None)
    entity_embeddings_df = entity_embeddings_df.drop(['vector'], axis = 1)

    X_0 = [str(user).lstrip("U") for user in behaviors_df['user_id']]

    X_1 = np.empty(50,dtype=object)
    X_2 = np.empty(50,dtype=object)
    X_3 = np.empty(50,dtype=object)
    X_4 = np.empty(50,dtype=object)
    X_5 = np.empty(50,dtype=object)



    counter =0
    news_ids = [None] * 50

    for h in behaviors_df['history']:
        his = []
 
        for i in str(h).split(' '):
                if(i[0] == 'N'):
                    temp = (news_df[news_df['id'] == i]['title_entities'])
                    q = temp.iloc[0][temp.iloc[0].find("WikidataId\": ")+len("WikidataId\": ")+1:temp.iloc[0].find("\", \"Confidence\"")]

                    RQs = str(entity_embeddings_df['id'][q])if q in entity_embeddings_df['id'].keys() else "0 0"
                    floats = [float(x) for x in RQs.split(" ") if is_float(x)]
                    RQs = sum(floats) / len(floats)
                    # print(f"in history {RQs}")

                    # print(temp.find("Q"))
                    # print(temp.find("WikidataId"))
                    his.append(RQs)
        # if(len(his) == 0):
        #     print("empty his")
        #     # news_ids.append(str(h).split(' ')[0])
        # else:
        #     print(len(his))
        #     # news_ids[coun] = (his)
        # coun = coun + 1
        X_1[counter] = his[:1][0]  if (len(his[:1])  != 0) else 0
        X_2[counter] = his[1:2][0] if (len(his[1:2])  != 0) else 0
        X_3[counter] = his[2:3][0] if (len(his[2:3])  != 0) else 0
        X_4[counter] = his[3:4][0] if (len(his[3:4])  != 0) else 0
        X_5[counter] = his[4:5][0] if (len(his[4:5])  != 0) else 0
        counter = counter +1


    indx = 0


   
 

    X_6 = np.empty(50,dtype=object)
    X_7 = np.empty(50,dtype=object)
    X_8 = np.empty(50,dtype=object)
    X_9 = np.empty(50,dtype=object)
    X_10 = np.empty(50,dtype=object)

    X_11 = np.empty(50,dtype=object)
    X_12 = np.empty(50,dtype=object)
    X_13 = np.empty(50,dtype=object)
    X_14 = np.empty(50,dtype=object)
    X_15 = np.empty(50,dtype=object)





    recommendation_newsIds = [None] * 50
    temp_imp = [None] * 50

    count = 0

    for h in behaviors_df['impressions']:
        rec = []
        imp = []
        for i in str(h).split(' '):

                # if(i.split('-')[1] == "1"):
                    temp = (news_df[news_df['id'] == i.split('-')[0]]['title_entities'])
                    if(len(temp) == 0):
                        q = "0"
                    else:
                        # print("found")
                        q = temp.iloc[0][temp.iloc[0].find("WikidataId\": ")+len("WikidataId\": ")+1:temp.iloc[0].find("\", \"Confidence\"")]

                    RQs = str(entity_embeddings_df['id'][q])if q in entity_embeddings_df['id'].keys() else "0 0"
                    floats = [float(x) for x in RQs.split(" ") if is_float(x)]
                    RQs = sum(floats) / len(floats)
                    # print(f"in rec {RQs}")

                    rec.append(RQs)
                    imp.append(i.split('-')[1])



        X_6[count] = rec[:1][0] if (len(rec[:1])  != 0) else 0
        X_7[count] = rec[1:2][0] if (len(rec[1:2])  != 0) else 0
        X_8[count] = rec[2:3][0] if (len(rec[2:3])  != 0) else 0
        X_9[count] = rec[3:4][0] if (len(rec[3:4])  != 0) else 0
        X_50[count] = rec[4:5][0] if (len(rec[4:5])  != 0) else 0

        X_11[count] = imp[:1][0] if (len(imp[:1])  != 0) else 0
        X_12[count] = imp[1:2][0] if (len(imp[1:2])  != 0) else 0
        X_13[count] = imp[2:3][0] if (len(imp[2:3])  != 0) else 0
        X_14[count] = imp[3:4][0] if (len(imp[3:4])  != 0) else 0
        X_15[count] = imp[4:5][0] if (len(imp[4:5])  != 0) else 0
        count = count +1












    # data = np.vstack([X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7,X_8, X_9, X_50,  X_11, X_12, X_13, X_14, X_15])
    data = np.vstack([X_0,X_1, X_6, X_11])

    data = np.transpose(data)
    # pd.DataFrame(data, columns = ['user_id','history','recommendation', 'impression']).to_csv('data.csv')
    print(data)

    return (data)

def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False
# get_data()

# def causal_structure(struct_data):
#     sm = from_pandas(struct_data)
    
#     viz = plot_structure(
#         sm,
#         graph_attributes={"scale": "0.5"},
#         all_node_attributes=NODE_STYLE.WEAK,
#         all_edge_attributes=EDGE_STYLE.WEAK,
#         prog='fdp',
#     )
#     Image(viz.draw(format='png'))


# def check_independence(data):
#     chi2_contingency(data.astype(np.int))[0:3]

# get_data()