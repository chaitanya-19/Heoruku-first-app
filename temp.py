# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC

data=load_iris()
target=data.target
x=data.data
df=pd.DataFrame(x,columns=data.feature_names)
df['target']=target

model=SVC()
model.fit(x,target)
print(model.score(x,target))