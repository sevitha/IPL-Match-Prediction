# data analysis and wrangling
import inline as inline
import matplotlib
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("./IPL_Matches_2008_2022.csv")
# remove unnecessary columns
data = data.drop(['ID'], axis=1)
data = data.drop(['City'], axis=1)
data = data.drop(['Date'], axis=1)
data = data.drop(['SuperOver'], axis=1)
data = data.drop(['method'], axis=1)
data = data.drop(['Player_of_Match'], axis=1)
data = data.drop(['Umpire1'], axis=1)
data = data.drop(['Umpire2'], axis=1)
data = data.drop(['Team1Players'], axis=1)
data = data.drop(['Team2Players'], axis=1)
# drop undecided matches
data['WinningTeam'] = data['WinningTeam'].dropna()
data.info()

# Map team names to numbers
# some teams have been renamed, so they map to the same value
# ex: "Punjab Kings" and "Kings XI Punjab"
teamMapping = {"Gujarat Titans": 0, "Rajasthan Royals": 1, "Royal Challengers Bangalore": 2,
               "Punjab Kings": 3, "Kings XI Punjab": 3, "Mumbai Indians": 4,
               "Lucknow Super Giants": 5, "Sunrisers Hyderabad": 6, "Delhi Capitals": 7,
               "Delhi Daredevils": 7, "Kolkata Knight Riders": 8, "Chennai Super Kings": 9,
               "Rising Pune Supergiant": 10, "Rising Pune Supergiants": 10, "Gujarat Lions": 11,
               "Pune Warriors": 12, "Deccan Chargers": 13, "Kochi Tuskers Kerala": 14}

data['Team1'] = data['Team1'].dropna().map(teamMapping).astype(int)
data['Team2'] = data['Team2'].dropna().map(teamMapping).astype(int)
data['TossWinner'] = data['TossWinner'].dropna().map(teamMapping).astype(int)
data['WinningTeam'] = data['WinningTeam'].dropna().map(teamMapping).astype(int)
# print(data['Team1'].unique())

# redefine 'WinningTeam' to be 1 if Team1 won and -1 if Team2 won
# print(data.head())
data.loc[data['WinningTeam'] == data['Team1'], ['WinningTeam']] = 1
data.loc[data['WinningTeam'] == data['Team2'], ['WinningTeam']] = -1
# print(data.head())

# print(data[['Team1', 'WinningTeam']].groupby(['Team1'],
#      as_index=False).mean().sort_values(by='WinningTeam', ascending=False))
# print(data[['Team2', 'WinningTeam']].groupby(['Team2'],
#      as_index=False).mean().sort_values(by='WinningTeam', ascending=False))

# use target encoding to deal with nominal data
winRateArray = []

for i in range(15):
    # calc number of won games
    numWonGames = (data.loc[(data['Team1'] == i) & (data['WinningTeam'] == 1)]['WinningTeam'].sum()
                 - data.loc[(data['Team2'] == i) & (data['WinningTeam'] == -1)]['WinningTeam'].sum())
    # print(numWonGames)

    # calc number of played games
    numPlayedGames = (data.loc[data['Team1'] == i].shape[0] + data.loc[data['Team2'] == i].shape[0])
    # print(numPlayedGames)

    # calc win rate
    winRate = numWonGames / numPlayedGames
    winRateArray.insert(i, winRate)

print(winRateArray)

winRateMap = {}

data['Team1'] = data['Team1'].map(lambda x: winRateArray[x])
data['Team2'] = data['Team2'].map(lambda x: winRateArray[x])

print(data.head())