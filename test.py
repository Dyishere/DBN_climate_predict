#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference
import pickle
from classifier import Classifier


# In[2]:


with open("./model/dbn2_3_24_2000.dat", 'rb') as f1:
	dbn = pickle.load(f1)

with open("./model/cls2_3_24_2000.dat", "rb") as f2:
	cls = pickle.load(f2)


# In[3]:


dbn_infer = DBNInference(dbn)


# In[4]:


data_path = "./jena_climate_2009_2016.csv"
dataset = pd.read_csv(data_path, parse_dates=['Date Time'], index_col=['Date Time'])


# In[5]:


num_train = 400000
num_test = 20451


# In[6]:


dataset_train = dataset[:num_train]
dataset_test = dataset[num_train:]

dataset_test


# In[7]:


data = dataset_test.values.T
data.shape


# In[8]:


windows = 3
sample = 24
scale = 2
cluster_v = [3, 3, 3, 5, 3, 3, 4, 2, 4, 4, 3, 2, 2, 4]
cluster_l = [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
cluster_v = [x * scale for x in cluster_v]


# In[11]:


lb = ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)']
params_label = []
pred_label = []
evi_label = []
for l in range(len(lb)):
    if cluster_l[l] == 1:
        params_label.append(lb[l])
        pred_label.append((lb[l], windows-1))
        evi_label.append((lb[l], 0))
        evi_label.append((lb[l], 1))
params_label


# In[12]:


_labels = cls.predict(lb[:-1], data)
_labels = np.array(_labels).T
_labels.shape


# In[13]:


labels = []
for i in range(len(cluster_l)):
	if cluster_l[i] == 1:
		labels.append(_labels[:, i])
labels = np.array(labels).T
labels.shape


# In[14]:


pred_data = np.zeros(shape=(labels.shape[0] - windows, labels.shape[1] * windows))
pred_data.shape


# In[15]:


for i in range(windows):
    pred_data[:, labels.shape[1] * i:labels.shape[1] * (i + 1)] = labels[i:-windows + i, :]
pred_data = np.array(pred_data[::sample], dtype=np.int_)
pred_data.shape


# In[ ]:


num_F = 0
for i in range(pred_data.shape[0]):
    evidence = {}
    for j in range(len(evi_label[0])):
        evidence[evi_label[j]] = pred_data[i, j]
    print(pred_data[i, len(evi_label[0]):])
    result = dbn_infer.query(variables=pred_label, evidence=evidence).values
    if result[i, j] != pred_data[i, len(evi_label[0])+j:
    	num_F += 1
print('Correct rate: {}'.format((pred_data.shape[0]-num_F)/pred_data.shape[0]))

