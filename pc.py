from itertools import combinations
from scipy.stats import norm
import numpy as np
import math
import pdb



def skeleton(cov, n_node,n_sample):

    C = np.ones((n_node,n_node))

    S = []
    for i in range(n_node):
        S.append([])
        for j in range(n_node):
            S[i].append([])

    pairs = []
    for i in range(n_node):
        for j in range(n_node - i):
            if(i != (n_node - j - 1)):  
                pairs.append((i, (n_node - j - 1)))
            else:
                C[i, i] = 0

    l = -1    
    while 1:
        l = l + 1
        flag = True   
        for (i, j) in pairs:

            adj_set = get_adjSet(i, C, n_node)    
            if(C[i][j] == 1) & (len(adj_set) >= l):    
                flag =False   
                adj_set.remove(j)    

                combin_set = combinations(adj_set, l)    
                for K in combin_set:
                    if fisher_z_test(i, j, list(K), cov, n_sample):
                        print("independent")   
                        C[i][j] = 0
                        C[j][i] = 0

                        S[i][j] = list(K)
                        S[j][i] = list(K)
                        break    
                    else:
                        continue
            else:
                continue

        if flag:
            break

    return C, S


def get_adjSet(i, G, n_node):
    adj = []
    for j in range(n_node):
        if G[i][j] == 1:
            adj.append(j)
    return adj




def direction(C, S):
    
    G = C
    n_node = C.shape[0]


    pairs = []
    for i in range(n_node):
        for j in range(n_node):
            if(i != j):    
                if(C[i][j] == 1):
                    pairs.append((i, j))
    
    triples = []
    for (i, j) in pairs:
        for k in range(n_node):
            if(C[j][k] == 1) & (k != i):
                triples.append([i, j, k])
    
    
    #  i-j-k， # i and k are not adjacent and (if and only if j is not in the sep_set (i,k)), then i -> j <- k
    for [i, j, k] in triples:
        if (G[i][j] == 1) & (G[j][i] == 1) & (G[k][j] == 1) & (G[j][k] == 1) & (G[i][k] == 0) & (
            G[k][i] == 0):    
            if j not in S[i][k]:
                G[j][i] = 0
                G[j][k] = 0

    #rule1: # for  i -> j - k  we have   i -> j -> k
    for [i, j, k] in triples:
        if (G[i][j] == 1) & (G[j][i] == 0) & (G[k][j] == 1) & (G[j][k] == 1) & (G[i][k] == 0) & (
            G[k][i] == 0):
            G[k][j] = 0

    #rule2: #  for  i -> j -> k and i-k  we have   i -> k
    for [i, j, k] in triples:
        if (G[i][j] == 1) & (G[j][i] == 0) & (G[j][k] == 1) & (G[k][j] == 0) & (G[i][k] == 1) & (
            G[k][i] == 1):
            G[k][i] = 0

    return G



def fisher_z_test(i, j, K, cov, n_sample):
    indep = True

    if len(K) == 0:
        r = cov[i, j]
    else:
        corr = cov[np.ix_([i] + [j] + K, [i] + [j] + K)]
        partial_corr = np.linalg.pinv(corr)   
        #-1*pr(x,y)/sqrt(pr(x,x)*pr(y,y))
        r = (-1 * partial_corr[0, 1]) / (math.sqrt(abs(partial_corr[0, 0] * partial_corr[1, 1])))
    
    r = min(0.99999, max(r, -0.99999))     

    #fisher z
    z = 0.5 * math.log1p((2 * r) / (1 - r))    
    z_standard = z * math.sqrt(n_sample - len(K) - 3)

    alpha = 0.005   
    if 2 * (1 - norm.cdf(abs(z_standard))) >= alpha:
        indep = True
    else:
        indep = False
    return indep
