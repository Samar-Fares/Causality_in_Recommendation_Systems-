import pandas as pd
from test import get_data

data= pd.read_csv('data.csv', sep=',')
data = data.drop("Unnamed: 0",axis=1)
users = []
for i in range(len(data)):
    if(float(data['history'][i]) <= -0.004 ):
        users.append(data['user_id'][i])
print((users))

test_data = get_data()
df = pd.DataFrame(test_data, columns = ['user_id','history','recom', 'imp'])
for i in range(len(df)):
    if (df['user_id'][i] in users):
        print('in')
        print(df['history'][i])
