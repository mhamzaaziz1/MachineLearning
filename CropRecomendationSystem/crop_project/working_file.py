# -*- coding: utf-8 -*-
"""
Created on Tue May  1 19:25:44 2018

@author: Hamza Aziz
"""
import numpy as np
import sys 
from tkinter import * 
import pandas as pd
import seaborn as s
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


features=pd.read_csv("C:/Users/Anderson/Downloads/NUSW-NB15_features.csv")
coloumns=features.Name
df=pd.read_csv("C:/Users/Anderson/Downloads/UNSW-NB15_1.csv", header=None, names=coloumns, low_memory=False)
newdf=df
def flatten(l):
    try:
        return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]
    except IndexError:
        return []

class UNSWNB15:
    def __init__(self):
        features=pd.read_csv("C:/Users/Anderson/Downloads/NUSW-NB15_features.csv")
        coloumns=features.Name
        df=pd.read_csv("C:/Users/Anderson/Downloads/UNSW-NB15_1.csv", header=None, names=coloumns, low_memory=False)
    
    def ShowDescription():
        root = Tk() 
        t1 = Text(root) 
        t1.pack() 
        class PrintToT1(object): 
         def write(self, s):
             t1.insert(END, s) 
        
        sys.stdout = PrintToT1() 
        print (features)

        
    
    def PreProcessing():
        df=pd.read_csv("C:/Users/Anderson/Downloads/UNSW-NB15_1.csv", header=None, names=coloumns, low_memory=False)
        root = Tk() 
        t1 = Text(root) 
        t1.pack() 
        class PrintToT1(object): 
         def write(self, s):
             t1.insert(END, s) 
        
        sys.stdout = PrintToT1() 
        print (df)
        
        plt.figure(figsize=(4,8))
        ax=s.countplot(x='Label',data=df )
        for p in ax.patches:
                ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()+50)) 
        
        root = Tk() 
        t1 = Text(root) 
        t1.pack() 
        class PrintToT1(object): 
         def write(self, s):
             t1.insert(END, s) 
    
        sys.stdout = PrintToT1() 
        print (df.describe().transpose())
        #mainloop()
    
                
        df=df.drop(['srcip', 'sport','dstip', 'dsport', 'attack_cat'], axis=1)
        df_categorical_values_enc=df.select_dtypes(include=['object']).apply(LabelEncoder().fit_transform)
                # proto
        proto=sorted(df.proto.unique())
        string1 = 'Proto_'
        proto_2=[string1 + x for x in proto]
        # service
        service=sorted(df.service.unique())
        string2 = 'service_'
        service_2=[string2 + x for x in service]
        # state
        state=sorted(df.state.unique())
        string3 = 'state_'
        state_2=[string3 + x for x in state]
        # put together
        dumcols=[]
        #dumcols=''.join(proto_2 + service_2 + state_2)
        dumcols.append(proto_2)
        dumcols.append(service_2)
        dumcols.append(state_2)
        
        dumcols=flatten(dumcols)
        
        enc = OneHotEncoder()
        df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
        df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)
        df_cat_data.head()
        
        newdf=df.join(df_cat_data)
        newdf.drop('state', axis=1, inplace=True)
        newdf.drop('proto', axis=1, inplace=True)
        newdf.drop('service', axis=1, inplace=True)
        
        x=df.drop(['Label'], axis=1)
        y=df.Label
        
        root = Tk() 
        t1 = Text(root) 
        t1.pack() 
        class PrintToT1(object): 
         def write(self, s):
             t1.insert(END, s) 
    
        sys.stdout = PrintToT1() 
        print (newdf)
        #mainloop()
    

    def RecursiveFeatureElimination():
        
        from sklearn.feature_selection import RFE
        df=pd.read_csv("C:/Users/Anderson/Downloads/UNSW-NB15_1.csv", header=None, names=coloumns, low_memory=False)
        df=df.drop(['srcip', 'sport','dstip', 'dsport', 'attack_cat'], axis=1)
        df_categorical_values_enc=df.select_dtypes(include=['object']).apply(LabelEncoder().fit_transform)
        # proto
        proto=sorted(df.proto.unique())
        string1 = 'Proto_'
        proto_2=[string1 + x for x in proto]
        # service
        service=sorted(df.service.unique())
        string2 = 'service_'
        service_2=[string2 + x for x in service]
        # state
        state=sorted(df.state.unique())
        string3 = 'state_'
        state_2=[string3 + x for x in state]
        # put together
        dumcols=[]
        #dumcols=''.join(proto_2 + service_2 + state_2)
        dumcols.append(proto_2)
        dumcols.append(service_2)
        dumcols.append(state_2)
        
        dumcols=flatten(dumcols)
        
        enc = OneHotEncoder()
        df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
        df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)
        df_cat_data.head()
        
        newdf=df.join(df_cat_data)
        newdf.drop('state', axis=1, inplace=True)
        newdf.drop('proto', axis=1, inplace=True)
        newdf.drop('service', axis=1, inplace=True)
        
        x=newdf.drop(['Label'], axis=1)
        y=newdf.Label
        
        clf = RandomForestClassifier(n_jobs=2)
        rfe = RFE(estimator=clf, n_features_to_select=13, step=1)
        rfe.fit(x, y)
        X_rfe=rfe.transform(x)
        true=rfe.support_
        rfecolindex=[i for i, x in enumerate(true) if x]
        rfecolname=list(colNames[i] for i in rfecolindex)
        
        
        root = Tk() 
        t1 = Text(root) 
        t1.pack() 
        class PrintToT1(object): 
         def write(self, s):
             t1.insert(END, s) 
    
        sys.stdout = PrintToT1() 
        print (rfecolname)
         
    def Model():
        from sklearn.model_selection import  train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.33, random_state=42)
        from sklearn.ensemble import RandomForestClassifier
        clf=RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                   max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                   bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                                   warm_start=False, class_weight=None)
        clf.fit(X_train,y_train)
    
    def TestModel():
        from sklearn.model_selection import  train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.33, random_state=42)
        from sklearn.ensemble import RandomForestClassifier
        clf=RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2,
                                   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                   max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                   bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                                   warm_start=False, class_weight=None)
        clf.fit(X_train,y_train)
        pd.crosstab(y_test, y_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])
        from sklearn.metrics import classification_report 
        
        root = Tk() 
        t1 = Text(root) 
        t1.pack() 
        class PrintToT1(object): 
         def write(self, s):
             t1.insert(END, s) 
    
        sys.stdout = PrintToT1() 
        print (pd.crosstab(y_test, y_pred, rownames=['Actual attacks'], colnames=['Predicted attacks']))
        print("Model evaluation\n"+classification_report(y_test,y_pred))
        mainloop()
    
    
    def RfeGraph():
        import matplotlib.pyplot as plt
        from sklearn.feature_selection import RFECV
        from sklearn.model_selection import StratifiedKFold
        
        # Create the RFE object and compute a cross-validated score.
        # The "accuracy" scoring is proportional to the number of correct
        # classifications
        rfecv = RFECV(estimator=clf, step=1, cv=10, scoring='accuracy')
        rfecv.fit(X_test, y_test)
        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.title('RFECV')
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()