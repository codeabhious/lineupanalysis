# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 08:34:23 2017

@author: Abhijit
"""
import projectlineups
from flask import Flask, render_template,request
import pandas as pd
app = Flask(__name__)
@app.route("/")
def main():
    return render_template("index.html")
 
@app.route('/predictions', methods = ['POST'])
def predictions():
    player1 = str(request.form["player1"].encode("ascii","ignore").decode("UTF-8"))
    player2 = str(request.form["player2"].encode("ascii","ignore").decode("UTF-8"))
    player3 = str(request.form["player3"].encode("ascii","ignore").decode("UTF-8"))
    player4 = str(request.form["player4"].encode("ascii","ignore").decode("UTF-8"))
    player5 = str(request.form["player5"].encode("ascii","ignore").decode("UTF-8"))
    fp =  "C:\\Users\\Abhijit\\Documents\\NylonCalc\\One Ball Rule\\clusterteamadv.csv"
    players = pd.DataFrame({"Player 1":[player1],"Player 2":[player2],"Player 3":[player3],
                            "Player 4":[player4],"Player 5":[player5]})
    projection = projectlineups.final_predict(fp,"PTS",players)[0]
    return render_template("predictions.html",player1=player1,player2=player2,player3=player3,
                           player4=player4,player5=player5,projection = projection)

if __name__ == "__main__":
    app.run()