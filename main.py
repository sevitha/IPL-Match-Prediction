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
data = data.drop(['WonBy'], axis=1)
data = data.drop(['Margin'], axis=1)
data = data.drop(['Season'], axis=1)
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
# print(data['Venue'].unique())

venueMapping = {'Narendra Modi Stadium, Ahmedabad': 1, 'Eden Gardens, Kolkata': 2,
                'Wankhede Stadium, Mumbai': 3, 'Brabourne Stadium, Mumbai': 4,
                'Dr DY Patil Sports Academy, Mumbai': 5,
                'Maharashtra Cricket Association Stadium, Pune': 6,
                'Dubai International Cricket Stadium': 7, 'Sharjah Cricket Stadium': 8,
                'Zayed Cricket Stadium, Abu Dhabi': 9, 'Arun Jaitley Stadium, Delhi': 10,
                'MA Chidambaram Stadium, Chepauk, Chennai': 11, 'Sheikh Zayed Stadium': 12,
                'Rajiv Gandhi International Stadium': 13,
                'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 14,
                'MA Chidambaram Stadium': 15, 'Punjab Cricket Association IS Bindra Stadium': 16,
                'Wankhede Stadium': 17, 'M.Chinnaswamy Stadium': 18, 'Arun Jaitley Stadium': 19,
                'Eden Gardens': 20, 'Sawai Mansingh Stadium': 21,
                'Maharashtra Cricket Association Stadium': 22, 'Holkar Cricket Stadium': 23,
                'Rajiv Gandhi International Stadium, Uppal': 24, 'M Chinnaswamy Stadium': 25,
                'Feroz Shah Kotla': 26, 'Green Park': 27,
                'Punjab Cricket Association IS Bindra Stadium, Mohali': 28,
                'Saurashtra Cricket Association Stadium': 29,
                'Shaheed Veer Narayan Singh International Stadium': 30,
                'JSCA International Stadium Complex': 31, 'Brabourne Stadium': 32,
                'Punjab Cricket Association Stadium, Mohali': 33,
                'MA Chidambaram Stadium, Chepauk': 34, 'Sardar Patel Stadium, Motera': 35,
                'Barabati Stadium': 36, 'Subrata Roy Sahara Stadium': 37,
                'Himachal Pradesh Cricket Association Stadium': 38,
                'Dr DY Patil Sports Academy': 39, 'Nehru Stadium': 40,
                'Vidarbha Cricket Association Stadium, Jamtha': 41, 'New Wanderers Stadium': 42,
                'SuperSport Park': 43, 'Kingsmead': 44, 'OUTsurance Oval': 45, "St George's Park": 46,
                'De Beers Diamond Oval': 47, 'Buffalo Park': 48, 'Newlands': 49}

data['Venue'] = data['Venue'].dropna().map(venueMapping).astype(int)

# use target encoding to deal with nominal data
winRateList = []
invertedTeamDict = dict(map(reversed, teamMapping.items()))

for i in range(15):
    # calc number of won games
    numWonGames = data.loc[data['WinningTeam'] == i]['WinningTeam'].shape[0]

    # calc number of played games
    numPlayedGames = (data.loc[data['Team1'] == i].shape[0] + data.loc[data['Team2'] == i].shape[0])
    #    print('=======')
    #    print(invertedTeamDict[i])
    #    print(numWonGames)
    #    print(numPlayedGames)
    #    print('=======')

    # calc win rate
    winRate = numWonGames / numPlayedGames
    winRateList.insert(i, winRate)

# print(winRateList)

# Some data exploration to see who won the most coin tosses

tossWinRateList = []
batPickRateList = []

for i in range(15):
    # calc number of won games
    numWonTosses = data.loc[data['TossWinner'] == i]['TossWinner'].shape[0]
    timesPickedBat = data.loc[(data['TossWinner'] == i) & (data['TossDecision'] == 'bat')]['TossWinner'].shape[0]
    timesPickedField = numWonTosses - timesPickedBat

    # calc number of played games
    numPlayedGames = (data.loc[data['Team1'] == i].shape[0] + data.loc[data['Team2'] == i].shape[0])

    # calc win rate
    tossWinRate = numWonTosses / numPlayedGames
    tossWinRateList.insert(i, tossWinRate)
    batPickRateList.insert(i, timesPickedBat / numWonTosses)

#    print('=-=-=-=-=')
#    print(invertedTeamDict[i])
#    print(tossWinRate)
#    print(timesPickedBat / numWonTosses)
#    print('=-=-=-=-=')

# print(tossWinRateList)
# print(batPickRateList)

# redefine 'WinningTeam' to be 1 if Team1 won and -1 if Team2 won
# print(data.head())
data.loc[data['WinningTeam'] == data['Team1'], ['WinningTeam']] = 1
data.loc[data['WinningTeam'] == data['Team2'], ['WinningTeam']] = -1
# do the same of toss winner
data.loc[data['TossWinner'] == data['Team1'], ['TossWinner']] = 1
data.loc[data['TossWinner'] == data['Team2'], ['TossWinner']] = -1

data['Team1'] = data['Team1'].map(lambda x: winRateList[x])
data['Team2'] = data['Team2'].map(lambda x: winRateList[x])

# print(data.head())

# print(data[['Team1', 'WinningTeam']].groupby(['Team1'],
#      as_index=False).mean().sort_values(by='WinningTeam', ascending=False))
# print(data[['Team2', 'WinningTeam']].groupby(['Team2'],
#      as_index=False).mean().sort_values(by='WinningTeam', ascending=False))

# data.loc[data['WonBy'] == 'Wickets', ['WonBy']] = 1
# data.loc[data['WonBy'] == 'Runs', ['WonBy']] = 0

# make match number all numeric
data.loc[(data['MatchNumber'] == 'Qualifier 1') |
         (data['MatchNumber'] == 'Qualifier'), ['MatchNumber']] = 71
data.loc[(data['MatchNumber'] == 'Eliminator') |
         (data['MatchNumber'] == 'Elimination Final'), ['MatchNumber']] = 72
data.loc[(data['MatchNumber'] == 'Qualifier 2') |
         (data['MatchNumber'] == '3rd Place Play-Off') |
         (data['MatchNumber'] == 'Semi Final'), ['MatchNumber']] = 73
data.loc[data['MatchNumber'] == 'Final', ['MatchNumber']] = 74

data.loc[(data['TossDecision'] == 'bat') | (data['TossDecision'] == 'Bat'), ['TossDecision']] = 1
data.loc[(data['TossDecision'] == 'field') | (data['TossDecision'] == 'Field'), ['TossDecision']] = 0

data = data.dropna()

# print(data['TossDecision'])

# 5-fold cross validation DT
split_data = np.array_split(data, 5)

decision_tree = DecisionTreeClassifier()
accumulated_decision_tree_accuracy = 0

for x in range(0, 5):
    # remove testing set from training data
    recombinedData = (pd.concat([data, split_data[x]])).drop_duplicates(keep=False).copy()
    # set current fold as testing set
    fiveFoldTest_Y = split_data[x]["WinningTeam"].copy()
    fiveFoldTest_X = split_data[x].drop("WinningTeam", axis=1).copy()
    # train tree
    fiveFoldTrain_X = recombinedData.drop("WinningTeam", axis=1).copy()
    fiveFoldTrain_Y = recombinedData["WinningTeam"].copy()
    decision_tree.fit(fiveFoldTrain_X, fiveFoldTrain_Y)
    Y_pred = decision_tree.predict(fiveFoldTest_X)
    # accumulate total prediction
    accumulated_decision_tree_accuracy += round(decision_tree.score(fiveFoldTest_X, fiveFoldTest_Y) * 100, 5)
# find arv prediction
print(accumulated_decision_tree_accuracy / 5)

# 5-fold cross validation RF
split_data = np.array_split(data, 5)

random_forest = RandomForestClassifier(n_estimators=100)
accumulated_RF_accuracy = 0


for x in range(0, 5):
    # remove testing set from training data
    recombinedData = (pd.concat([data, split_data[x]])).drop_duplicates(keep=False).copy()
    # set current fold as testing set
    fiveFoldTest_Y = split_data[x]["WinningTeam"].copy()
    fiveFoldTest_X = split_data[x].drop("WinningTeam", axis=1).copy()
    # train tree
    fiveFoldTrain_X = recombinedData.drop("WinningTeam", axis=1).copy()
    fiveFoldTrain_Y = recombinedData["WinningTeam"].copy()
    random_forest.fit(fiveFoldTrain_X, fiveFoldTrain_Y)
    Y_pred = random_forest.predict(fiveFoldTest_X)
    # accumulate total prediction
    accumulated_RF_accuracy += round(random_forest.score(fiveFoldTest_X, fiveFoldTest_Y) * 100, 5)
# find arv prediction
print(accumulated_RF_accuracy / 5)
