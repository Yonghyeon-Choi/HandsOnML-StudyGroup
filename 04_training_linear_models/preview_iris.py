import pandas as pd
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
list(iris.keys())

print(iris)

data = pd.DataFrame(data= np.c_[iris['data'], iris['target']], columns= iris['feature_names'] + ['target'])
data['target'] = data['target'].map({0: "setosa", 1: "versicolor", 2: "virginica"})

data.to_csv("easily_recog_iris.csv", sep=',', na_rep='NaN')
data.to_csv("more_easily_recog_iris.tsv", sep='\t', na_rep='NaN')