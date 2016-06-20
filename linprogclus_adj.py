from pulp import *
from promethee import *
from adjcluster import *
import time
import csv
import pandas as pd
from scipy.cluster.vq import kmeans, vq
from statsmodels.sandbox.tools.tools_pca import pcasvd, pca
import matplotlib.pyplot as plt
from pca_kmeans_biplot import *

# criteria = ['Infrastructure', 'Spatial', 'Education', 'CultureEnv', 'Healthcare', 'Stability']
#
# weights = [0.15, 0.25, 0.075, 0.1875, 0.15, 0.1875]
#
# fctPrefCrit = [PreferenceType2(0), PreferenceType2(0), PreferenceType2(0),  PreferenceType2(0), PreferenceType2(0), PreferenceType2(0)]
#
# names = ['Hong Kong', 'Stockholm', 'Rome', 'New York', 'Atlanta', 'Buenos Aires', 'Santiago', 'Sao Paulo', 'Mexico City', 'New Delhi', 'Istanbul', 'Jakarta', 'Tehran', 'Dakar']
#
# eval_table = [[96.4, 75.0, 100.0, 85.9, 87.5, 95.0],
# [96.4, 58.9, 100.0, 91.2, 95.8, 95.0],
# [92.9, 67.3, 100.0, 91.7, 87.5, 80.0],
# [89.3, 65.2, 100.0, 91.7, 91.7, 70.0],
# [92.9, 42.9, 100.0, 91.7, 91.7, 85.0],
# [85.7, 42.3, 100.0, 85.9, 87.5, 70.0],
# [85.7, 35.1, 83.3, 89.1, 70.8, 75.0],
# [66.1, 52.4, 66.7, 80.3, 70.8, 60.0],
# [46.4, 65.8, 75.0, 82.4, 66.7, 45.0],
# [58.9, 58.6, 75.0, 55.6, 58.3, 55.0],
# [67.9, 47.5, 58.3, 68.8, 50.0, 55.0],
# [57.1, 42.3, 66.7, 59.3, 45.8, 50.0],
# [33.9, 53.6, 50.0, 35.9, 62.5, 50.0],
# [37.5, 22.6, 50.0, 59.7, 41.7, 50.0]]

# criteria = ['1', '2', '3', '4']
# weights = [0.25, 0.25, 0.25, 0.25]
# fctPrefCrit = [PreferenceType2(0), PreferenceType2(0), PreferenceType2(0),  PreferenceType2(0)]#
# eval_table = []
# with open('flower.csv', newline='') as csvfile:xx
#     content = csv.reader(csvfile, delimiter=',', quotechar='|')
#     for row in content:
#         eval_table.append(list(map(lambda x: float(x), row)))

criteria = ['crit1', 'crit2', 'crit3', 'crit4', 'crit5', 'crit6']
weights = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
fctPrefCrit = [PreferenceType2(0), PreferenceType2(0), PreferenceType2(0),  PreferenceType2(0), PreferenceType2(0), PreferenceType2(0)]
eval_table = []
with open('shanghai.csv', newline='') as csvfile:
    content = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in content:
        eval_table.append(list(map(lambda x: float(x), row)))
eval_table = eval_table[:]

# criteria = ['1', '2']
# weights = [0.5, 0.5]
# fctPrefCrit = [PreferenceType2(0), PreferenceType2(0)]
# eval_table = [[0, 0],
# [0.1, 0.1],
# [0.2, 0.2],
# [4, 4],
# [4.1, 4.1],
# [4.2, 4.2],
# [10, 10],
# [10.1, 10.1],
# [10.2, 10.2]]

uninetflows = uninetflows_eval(eval_table,criteria,weights,fctPrefCrit)
netflows = rankingP2(eval_table, criteria, weights, fctPrefCrit)

# uniposflows, uninegflows = uniflows_eval(candidats, criteres, init_weights, fctPrefCrit)

def netflow_eval(i,n,weights,gamma):
    return posflow_eval(i,n,weights,gamma) - negflow_eval(i,n,weights,gamma)

def posflow_eval(i,n,weights,gamma):
    crits = len(weights)
    posflow = 0
    for j in range(n):
        if j != i:
            for k in range(crits):
                posflow += weights[k] * gamma[(i,j,k)]
    return (1/(n-1)) * posflow

def negflow_eval(i,n,weights,gamma):
    crits = len(weights)
    negflow = 0
    for j in range(n):
        if j != i:
            for k in range(crits):
                negflow += weights[k] * gamma[(j,i,k)]
    return (1/(n-1)) * negflow

# Problem definition
prob = LpProblem("LPC",LpMaximize)

# Constants
M = 1e6
M2 = 1e2
N = 3 # Number of clusters
EPS = 1e-6
#init_weights = [0.1, 0.3, 0.6]
#criteria = [str(x) for x in range(len(init_weights))]
n = len(eval_table)
m = len(eval_table[0])
alts = range(n)
crits = range(m)
clusts = range(N)

# Variables
a_comb = []
gamma_comb = []
for i in alts:
    for j in alts:
        if i != j:
            a_comb.append((i,j))
            for k in crits:
                gamma_comb.append((i,j,k))

# alpha = LpVariable.dicts("alpha",a_comb,cat="Binary")
a = LpVariable.dicts("z",a_comb,cat="Binary")
# gamma = LpVariable.dicts("gamma",gamma_comb,cat="Binary")
beta1 = LpVariable.dicts("beta1",gamma_comb,cat="Binary")
beta2 = LpVariable.dicts("beta2",gamma_comb,cat="Binary")

# Objective function
# obj_hom = []
# obj_het = []
obj = []
gamma = []
for i in alts:
    gamma.append([[] for j in alts])
    for j in alts:
        if i != j:
            gamma[i][j] = [0 for k in crits]
            for k in crits:
                if eval_table[i][k] > eval_table[j][k]:
                    gamma[i][j][k] = 1
                # obj.append((2*beta1[(i,j,k)] + beta2[(i,j,k)] - gamma[(i,j,k)]) * weights[k])
                # obj.append((gamma[(i,j,k)]-beta1[(i,j,k)]-beta2[(i,j,k)])*weights[k])
                # obj.append(beta1[(i,j,k)] * weights[k])
                # obj.append((1-a[(i,j)]-a[(j,i)])*gamma[i][j][k]*weights[k])
                obj.append(0.55*a[(i,j)]*gamma[i][j][k]*weights[k] + 0.45*(1-a[(i,j)]-a[(j,i)])*gamma[i][j][k]*weights[k])

prob += lpSum(obj)

# Constraints
alpha = []
for i in alts:
    alpha.append([0 for j in alts])

    for j in alts:
        if i != j:

            prob += a[(i,j)] + a[(j,i)] <= 1
            # for k in crits:
                # prob += gamma[(i,j,k)] >= (eval_table[i][k]-eval_table[j][k])/M
                # prob += gamma[(i,j,k)] <= (eval_table[i][k]-eval_table[j][k])/M + 1 - EPS
                # prob += beta1[(i,j,k)] >= a[(i,j)] + gamma[(i,j,k)] - 1
                # prob += beta1[(i,j,k)] <= 0.5*(a[(i,j)] + gamma[(i,j,k)])
                # prob += beta2[(i,j,k)] >= a[(j,i)] + gamma[(i,j,k)] - 1
                # prob += beta2[(i,j,k)] <= 0.5*(a[(j,i)] + gamma[(i,j,k)])

                # prob += beta1[(i,j,k)] >= a[(i,j)] + gamma[i][j][k] - 1
                # prob += beta1[(i,j,k)] <= 0.5*(a[(i,j)] + gamma[i][j][k])
                # prob += beta2[(i,j,k)] >= a[(j,i)] + gamma[i][j][k] - 1
                # prob += beta2[(i,j,k)] <= 0.5*(a[(j,i)] + gamma[i][j][k])

            if netflows[i] > netflows[j]:
                alpha[i][j] = 1

            prob += a[(i,j)] <= alpha[i][j]
            # prob += alpha[i][j] + a[(i,j)] <= 2


#print(prob)

prob.writeLP("linprogclust_adj.lp")
start_time = time.time()
prob.solve(GUROBI())
stop_time = time.time() - start_time

for v in prob.variables():
    if 'z' in v.name:
        print(v.name, "=", v.varValue)
print("Objective function:", value(prob.objective))
print("Status:", LpStatus[prob.status])
print("Time:", stop_time)

adj = [[0 for i in alts] for i in alts]
print("Adjacency matrix")
for i in alts:
    for j in alts:
        if i != j:
            adj[i][j] = int(a[(i,j)].varValue)
    print(adj[i])

clust_repart = adjToCluster(adj)
print("Cluster repartition:",clust_repart)
f = open("clustering.m",'w')
f.write("clusts = [ ")
for val in clust_repart:
    f.write(str(val) + '\n')
f.write("];\n")
f.write("criteria = {")
for crit in criteria:
    f.write("'" + crit + "' ")
f.write('};')
f.close()

# clust_repart = []
# f = open('clustering.m','w')
# f.write("clusts = [ ")
# for i in alts:
#     for h in clusts:
#         if c[(i,h)].varValue == 1:
#             clust_repart.append(h)
#             f.write(str(h) + '\n')
# f.write("];\n")
# f.write("criteria = {")
# for crit in criteria:
#     f.write("'" + crit + "' ")
# f.write('};')
# f.close()
#
uninetflows.insert(0,criteria)
with open("uninetflows.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(uninetflows)

df = pd.io.parsers.read_csv('uninetflows.csv')
data = df[criteria]
# data = (data - data.mean()) / data.std()
pca = pcasvd(data, keepdim=0, demean=False)
colors = ['gbyrkgbyrkgbyrk'[i] for i in clust_repart]
plt.figure(1)
biplot(plt, pca, labels=data.index, colors=colors, xpc=1, ypc=2)
plt.show()

# iter = 0
# sols = []
# while iter < 5:
#     prob.solve(pulp.GLPK())
#     print(LpStatus[prob.status])
#     sols.append(prob.variables())
#
#     iter += 1
