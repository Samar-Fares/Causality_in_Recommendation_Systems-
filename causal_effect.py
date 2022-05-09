import graphviz
import pygraphviz
import pandas as pd
# Commented out IPython magic to ensure Python compatibility.
# Main imports
from econml.dml import DML, LinearDML, SparseLinearDML, CausalForestDML
from econml.cate_interpreter import SingleTreeCateInterpreter, \
                                    SingleTreePolicyInterpreter
# Helper imports
import numpy as np
from itertools import product
from sklearn.linear_model import (Lasso, LassoCV, LogisticRegression,
                                  LogisticRegressionCV,LinearRegression,
                                  MultiTaskElasticNet,MultiTaskElasticNetCV)
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
import imp 
import dowhy 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import pygraphviz 
import imp 
import dowhy 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import pygraphviz 

# Generic ML imports
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures

causal_graph = """digraph { 
user_id[label="User"]; 
history[label="History"]; 
recommendation[label="Recommendation"]; 
impression[label="Clicked"]; 
user_id -> impression; 
recommendation -- history;  
history -> impression; 
recommendation -> impression; 
}""" 
 
data= pd.read_csv('data.csv', sep=',')
data = data.drop("Unnamed: 0",axis=1)
model= dowhy.CausalModel( 
        data = data, 
        graph=causal_graph.replace("\n", " "), 
        treatment="recommendation", 
        outcome='impression'
        ) 
model.view_model() 
from IPython.display import Image, display 
display(Image(filename="causal_model.png")) 
 
#Identify the causal effect 
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True) 
print(identified_estimand) 
 
estimate = model.estimate_effect(identified_estimand, 
                                 method_name="backdoor.linear_regression",target_units="ate") 
# ATE = Average Treatment Effect 
print(estimate)

refute2_results=model.refute_estimate(identified_estimand, estimate,
        method_name="placebo_treatment_refuter")
print(refute2_results)

# Define estimator inputs
T = data['recommendation'] # intervention, or treatment
Y = data['impression'] # outcome of interest
X_data = data.drop(columns=['recommendation', 'impression']) # features

X_data

feature_names = X_data.columns.tolist()
# Define nuissance estimators
lgb_T_XZ_params = {
    'objective' : 'binary',
    'metric' : 'auc',
    'learning_rate': 0.1,
    'num_leaves' : 30,
    'max_depth' : 5
}

lgb_Y_X_params = {
    'metric' : 'rmse',
    'learning_rate': 0.1,
    'num_leaves' : 30,
    'max_depth' : 5
}
model_T_XZ = lgb.LGBMClassifier(**lgb_T_XZ_params)
model_Y_X = lgb.LGBMRegressor(**lgb_Y_X_params)
flexible_model_effect = lgb.LGBMRegressor(**lgb_Y_X_params)


test_customers = X_data.iloc[:1000]

# %matplotlib inline
est = LinearDML(model_y=RandomForestRegressor(),
                model_t=RandomForestRegressor(),
                random_state=123)
est.fit(data['impression'], data['recommendation'], X=X_data)
intrp = SingleTreeCateInterpreter(include_model_uncertainty=True, max_depth=2, min_samples_leaf=10)
intrp.interpret(est, test_customers)
plt.figure(figsize=(25, 5))
intrp.plot(feature_names=data['recommendation'], fontsize=12)

intrp = SingleTreePolicyInterpreter(risk_level=0.05, max_depth=2, min_samples_leaf=10)
intrp.interpret(est, test_customers, sample_treatment_costs=0.2)
plt.figure(figsize=(25, 5))
intrp.plot(feature_names=X_data.columns, fontsize=12)