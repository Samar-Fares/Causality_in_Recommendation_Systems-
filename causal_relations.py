from movielens_utils import get_data
import numpy as np
import pdb
import cdt
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci

from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.search.ScoreBased.GES import ges
import io
import matplotlib.image as mpimg




import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'



data = get_data()
# print (data.shape[0])
# # cov = np.cov(data)
# cg = pc(data.astype(np.float))
# print(f"type    {type(cg)}")

# cg.draw_pydot_graph()
# pdy = GraphUtils.to_pydot(cg)
# pdy.write_png('simple_test.png')

Record = ges(data.astype(np.float), score_func = "local_score_CV_multi", parameters = {
  "kfold": 10,
  "lambda": 0.1,
  "dlabel": [0, 1,2]
#   [1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]]
})
# Record['G'].draw_pydot_graph()
# print(f"type    {type(Record['G'].graph)}")
# print(Record['G'].get_node_names())
# print(Record)

pyd = GraphUtils.to_pydot(Record['G'])
tmp_png = pyd.create_png(f="png")
fp = io.BytesIO(tmp_png)
img = mpimg.imread(fp, format='png')
plt.axis('off')
plt.imshow(img)
plt.show()

print(Record)
