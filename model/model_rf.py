import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('pronostico_dataset.csv',sep=';')
df.loc[df.prognosis=='retinopathy','prognosis']=1
df.loc[df.prognosis=='no_retinopathy','prognosis']=0
df.drop(['ID'],axis=1,inplace=True)
x=df.iloc[:,:-1]
ss=StandardScaler()
x=ss.fit_transform(x)
y=df.iloc[:,-1]
y=y.astype(int)

model_rf = RandomForestClassifier(n_estimators=107,max_depth=2, max_features=4,criterion='gini',random_state=23)
model_rf.fit(x,y)

filename = 'model_final.sav'
filename_scaler='scaler.sav'
joblib.dump(model_rf,filename)
joblib.dump(ss,filename_scaler)