#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#for windows, download the zip file of graphviz.
#https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft
#https://graphviz.gitlab.io/_pages/Download/Download_windows.html
#then replace the path where you put it.
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[2]:


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
in_file = 'A:\Aadharsh\Repo\Bayesian-Classifier\COVID-19_formatted_dataset.xlsx'
dataset = pd.read_excel(in_file)
print('read success')
print(dataset['SARS-Cov-2 exam result'])
dataset["SARS-Cov-2 exam result"].replace({"negative": 0, "positive": 1}, inplace=True)
target = dataset['SARS-Cov-2 exam result']
print(target)
dataset.drop("SARS-Cov-2 exam result", axis=1,inplace=True)


# In[3]:


cols = cols = list(np.array([1]))+list(np.arange(3,17))
x = dataset.iloc[:,cols]
y = np.expand_dims(target.to_numpy(),axis=1)

x_train, x_test, y_train, y_test = train_test_split(x.to_numpy(), y, test_size=0.2, random_state=0)
print(len(y_train))


# In[4]:


clf = tree.DecisionTreeClassifier(max_depth =4, random_state=0)
clf = clf.fit(x_train, y_train)
feature_names={0:'age',1:'hematocrit',2:'hemo',3:'platelets',4:'mean_platelet_volume',5:'rbc',6:'lymphocytes',7:'MCHC',8:'leukocytes',9:'basophils',10:'MCH',11:'eosinophils',12:'MCV',13:'monocytes',14:'rdw'}
c_names={0: 'healthy', 1: 'covid_19'}
tree.plot_tree(clf, feature_names=feature_names, class_names=c_names)
plt.show()
dot_data = tree.export_graphviz(clf, out_file=None, class_names=c_names, feature_names=feature_names)
graph = graphviz.Source(dot_data)
graph.render("covid_tree") #output will be a pdf with the tree in high quality. plt.show will make plot blurry.


# In[5]:


y_pred = clf.predict(x_test)
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = clf.predict(x_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for DT = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for DT = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))
print(cm_test)

x_patient = x_test[0].reshape(1,-1)
y_out = clf.predict_proba(x_patient.reshape(1,-1))
print('predicted first test sample')
print(y_out)
y_out = clf.predict_proba(x_test)
print(y_out[0:4])


# In[6]:


n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
print(target)
dataset = x_train
target = y_train
x = dataset[:,[0,1]]
y = y = target
clf = DecisionTreeClassifier(max_depth=5).fit(x, y)
feature_names={0:'age',1:'hematocrit'}
c_names = {0:'healthy', 1:'covid'}
plot_tree(clf, filled=True,class_names=c_names)
plt.show()
plt.figure()
dot_data = tree.export_graphviz(clf, out_file=None, class_names=c_names,filled=True, feature_names=feature_names)
graph = graphviz.Source(dot_data)
graph.render("covid_tree_decision_boundary") #output will be a pdf with the tree in high quality. plt.show will make plot blurry.


# In[7]:


x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
cs = plt.contourf(xx, yy, z, cmap=plt.cm.RdYlBu)

plt.xlabel('x[0]')
plt.ylabel('x[1]')

for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(y == i)
    plt.scatter(x[idx, 0], x[idx, 1], c=color,cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.title("Decision surface using feature 0 and 1")
#plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()


# In[ ]:




