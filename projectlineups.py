# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 22:48:56 2017

@author: Abhijit
"""

import pandas as pd
import numpy as np
#data = pd.read_csv("C:\\Users\\Abhijit\\Documents\\NylonCalc\\One Ball Rule\\clusterteamadv.csv")
#projections = pd.read_csv("C:\\Users\\Abhijit\\Documents\\NylonCalc\\One Ball Rule\\lineuptoview.csv")
def get_cluster(projections,fp):
    #data = data[(data["Year"]==2017) | (data["Year"]==2016)] ### get this year
    initial = np.zeros((projections.shape[0],10))
    data = pd.read_csv(fp)
    for i in range(1,6):
        name = "Player " + str(i) ### get name
        playerdata = projections[name]
        pc = []
        for player in playerdata:
            pdata = data[data["Player ID"]==player]["Cluster"].values[0]
            pc.append(pdata)
        for j in range(0,10):
            if j not in pc:
                pc.append(j)
        lendiff = len(pc) - projections.shape[0]
        df = pd.get_dummies(pc).as_matrix()
        if lendiff > 0:
            df = df[0:projections.shape[0],:]
        initial = df + initial
    return initial/5
prediction = "PTS"
def load_models(prediction):
    import keras
    import pickle
    rfr = pickle.load(open("C:\\Users\\Abhijit\\Documents\\NylonCalc\\One Ball Rule\\predict"+prediction+"rfr.sav","rb"))
    svr = pickle.load(open("C:\\Users\\Abhijit\\Documents\\NylonCalc\\One Ball Rule\\predict"+prediction+"svr.sav","rb"))
    knr = pickle.load(open("C:\\Users\\Abhijit\\Documents\\NylonCalc\\One Ball Rule\\predict"+prediction+"knr.sav","rb"))
    eln = pickle.load(open("C:\\Users\\Abhijit\\Documents\\NylonCalc\\One Ball Rule\\predict"+prediction+"eln.sav","rb"))
    nn2 = keras.models.load_model("C:\\Users\\Abhijit\\Documents\\NylonCalc\\One Ball Rule\\predict"+prediction+".h5")
    return [rfr,svr,eln,knr,nn2]
def ensemble_predict(models,inputs):
    s = 0
    for model in models:
        s += np.reshape(model.predict(inputs),(inputs.shape[0],))/5
    return s

def final_predict(fp,prediction,players):
    models = load_models(prediction)
    inputs = get_cluster(players,fp)
    return ensemble_predict(models,inputs)
    
def getplayers(fp):
    return list(pd.read_csv(fp)["Player ID"].values)
    
        