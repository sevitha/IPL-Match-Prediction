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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
# data = data.drop(['Season'], axis=1)

# make season numeric only
data.loc[(data['Season'] == '2020/21'), ['Season']] = '2020'
data.loc[(data['Season'] == '2009/10'), ['Season']] = '2009'
data.loc[(data['Season'] == '2007/08'), ['Season']] = '2007'

data['Season'] = pd.to_numeric(data['Season'], 'coerce', 'integer')

# print(data['Season'].unique())

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

# map venue to most recent home team number, venues with no home team map to 0
venueMapping = {'Narendra Modi Stadium, Ahmedabad': 1, 'Eden Gardens, Kolkata': 8,
                'Wankhede Stadium, Mumbai': 4, 'Brabourne Stadium, Mumbai': 102,
                'Dr DY Patil Sports Academy, Mumbai': 12,
                'Maharashtra Cricket Association Stadium, Pune': 9,
                'Dubai International Cricket Stadium': 0, 'Sharjah Cricket Stadium': 0,
                'Zayed Cricket Stadium, Abu Dhabi': 0, 'Arun Jaitley Stadium, Delhi': 7,
                'MA Chidambaram Stadium, Chepauk, Chennai': 9, 'Sheikh Zayed Stadium': 0,
                'Rajiv Gandhi International Stadium': 100,
                'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 105,
                'MA Chidambaram Stadium': 9, 'Punjab Cricket Association IS Bindra Stadium': 3,
                'Wankhede Stadium': 4, 'M.Chinnaswamy Stadium': 2, 'Arun Jaitley Stadium': 7,
                'Eden Gardens': 8, 'Sawai Mansingh Stadium': 1,
                'Maharashtra Cricket Association Stadium': 9, 'Holkar Cricket Stadium': 104,
                'Rajiv Gandhi International Stadium, Uppal': 6, 'M Chinnaswamy Stadium': 2,
                'Feroz Shah Kotla': 0, 'Green Park': 11,
                'Punjab Cricket Association IS Bindra Stadium, Mohali': 3,
                'Saurashtra Cricket Association Stadium': 11,
                'Shaheed Veer Narayan Singh International Stadium': 0,
                'JSCA International Stadium Complex': 107, 'Brabourne Stadium': 1,
                'Punjab Cricket Association Stadium, Mohali': 3,
                'MA Chidambaram Stadium, Chepauk': 9, 'Sardar Patel Stadium, Motera': 0,
                'Barabati Stadium': 103, 'Subrata Roy Sahara Stadium': 106,
                'Himachal Pradesh Cricket Association Stadium': 3,
                'Dr DY Patil Sports Academy': 101, 'Nehru Stadium': 14,
                'Vidarbha Cricket Association Stadium, Jamtha': 13, 'New Wanderers Stadium': 0,
                'SuperSport Park': 0, 'Kingsmead': 0, 'OUTsurance Oval': 0, "St George's Park": 0,
                'De Beers Diamond Oval': 0, 'Buffalo Park': 0, 'Newlands': 0}

# map venue to most recent home team number, venues with no home team map to 0
venueMappingLinear = {'Narendra Modi Stadium, Ahmedabad': 1, 'Eden Gardens, Kolkata': 2,
                'Wankhede Stadium, Mumbai': 3, 'Brabourne Stadium, Mumbai': 4,
                'Dr DY Patil Sports Academy, Mumbai': 5,
                'Maharashtra Cricket Association Stadium, Pune': 6,
                'Dubai International Cricket Stadium': 7, 'Sharjah Cricket Stadium': 8,
                'Zayed Cricket Stadium, Abu Dhabi': 9, 'Arun Jaitley Stadium, Delhi': 10,
                'MA Chidambaram Stadium, Chepauk, Chennai': 11, 'Sheikh Zayed Stadium': 12,
                'Rajiv Gandhi International Stadium': 13,
                'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 14,
                'MA Chidambaram Stadium': 15, 'Punjab Cricket Association IS Bindra Stadium': 16,
                'Wankhede Stadium': 16, 'M.Chinnaswamy Stadium': 17, 'Arun Jaitley Stadium': 18,
                'Eden Gardens': 19, 'Sawai Mansingh Stadium': 20,
                'Maharashtra Cricket Association Stadium': 21, 'Holkar Cricket Stadium': 22,
                'Rajiv Gandhi International Stadium, Uppal': 23, 'M Chinnaswamy Stadium': 24,
                'Feroz Shah Kotla': 25, 'Green Park': 26,
                'Punjab Cricket Association IS Bindra Stadium, Mohali': 27,
                'Saurashtra Cricket Association Stadium': 28,
                'Shaheed Veer Narayan Singh International Stadium': 29,
                'JSCA International Stadium Complex': 30, 'Brabourne Stadium': 31,
                'Punjab Cricket Association Stadium, Mohali': 33,
                'MA Chidambaram Stadium, Chepauk': 34, 'Sardar Patel Stadium, Motera': 35,
                'Barabati Stadium': 36, 'Subrata Roy Sahara Stadium': 37,
                'Himachal Pradesh Cricket Association Stadium': 38,
                'Dr DY Patil Sports Academy': 39, 'Nehru Stadium': 40,
                'Vidarbha Cricket Association Stadium, Jamtha': 42, 'New Wanderers Stadium': 43,
                'SuperSport Park': 44, 'Kingsmead': 45, 'OUTsurance Oval': 46, "St George's Park": 47,
                'De Beers Diamond Oval': 48, 'Buffalo Park': 49, 'Newlands': 50}

data['Venue'] = data['Venue'].dropna().map(venueMapping).astype(int)
print(data.head())

# set venue to +1 if Team1 is the home team, -1 if team 2 is the home team or, 0 otherwise
data.loc[(data['Venue'] == 100) & (data['Season'] >= 2012), ['Venue']] = 13
data.loc[(data['Venue'] == 100) & (data['Season'] < 2012), ['Venue']] = 6

data.loc[(data['Venue'] == 101) & (data['Season'] <= 2008), ['Venue']] = 4
data.loc[(data['Venue'] == 101) & (data['Season'] > 2008) & (data['Season'] < 2011), ['Venue']] = 13
data.loc[(data['Venue'] == 101) & (data['Season'] >= 2011), ['Venue']] = 12

data.loc[(data['Venue'] == 102) & (data['Season'] <= 2010), ['Venue']] = 4
data.loc[(data['Venue'] == 102) & (data['Season'] > 2010), ['Venue']] = 1

# this was shared by 2 teams in 2014
data.loc[(data['Venue'] == 103) & (data['Season'] <= 2012), ['Venue']] = 4
data.loc[(data['Venue'] == 103) & (data['Season'] > 2012), ['Venue']] = 3

data.loc[(data['Venue'] == 104) & (data['Season'] <= 2011), ['Venue']] = 14
data.loc[(data['Venue'] == 104) & (data['Season'] > 2011), ['Venue']] = 3

data.loc[(data['Venue'] == 105) & (data['Season'] < 2015), ['Venue']] = 13
data.loc[(data['Venue'] == 105) & (data['Season'] == 2015), ['Venue']] = 6
data.loc[(data['Venue'] == 105) & (data['Season'] > 2015), ['Venue']] = 4

data.loc[(data['Venue'] == 106) & (data['Season'] < 2015), ['Venue']] = 12
data.loc[(data['Venue'] == 106) & (data['Season'] == 2015), ['Venue']] = 3
data.loc[(data['Venue'] == 106) & (data['Season'] > 2015) & (data['Season'] <= 2017), ['Venue']] = 4
data.loc[(data['Venue'] == 106) & (data['Season'] > 2017), ['Venue']] = 9

data.loc[(data['Venue'] == 107) & (data['Season'] == 2013), ['Venue']] = 8
data.loc[(data['Venue'] == 107) & (data['Season'] == 2014), ['Venue']] = 9

data.loc[(data['Venue'] > 99), ['Venue']] = 0

data.loc[data['Venue'] == data['Team1'], ['Venue']] = 1
data.loc[data['Venue'] == data['Team2'], ['Venue']] = -1
data.loc[(data['Venue'] != data['Team1']) & (data['Venue'] != data['Team2']), ['Venue']] = 0

# data = data.drop(['Season'], axis=1)

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

#   print('=-=-=-=-=')
#   print(invertedTeamDict[i])
#   print(tossWinRate)
#   print(timesPickedBat / numWonTosses)
#   print('=-=-=-=-=')

#   print(tossWinRateList)
#   print(batPickRateList)

# redefine 'WinningTeam' to be 1 if Team1 won and -1 if Team2 won
data.loc[data['WinningTeam'] == data['Team1'], ['WinningTeam']] = 1
data.loc[data['WinningTeam'] == data['Team2'], ['WinningTeam']] = -1
# do the same of toss winner
data.loc[data['TossWinner'] == data['Team1'], ['TossWinner']] = 1
data.loc[data['TossWinner'] == data['Team2'], ['TossWinner']] = -1

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

# k-fold cross validation

def k_fold_cross_validation(k, learner, input_data):
    split_data = np.array_split(input_data, k)
    accumulated_accuracy = 0

    for x in range(0, k):
        # remove testing set from training data
        recombined_data = (pd.concat([data, split_data[x]])).drop_duplicates(keep=False).copy()
        # set current fold as testing set
        test_y = split_data[x]["WinningTeam"].copy()
        test_x = split_data[x].drop("WinningTeam", axis=1).copy()
        # train tree
        train_x = recombined_data.drop("WinningTeam", axis=1).copy()
        train_y = recombined_data["WinningTeam"].copy()
        learner.fit(train_x, train_y)
        # accumulate total prediction
        accumulated_accuracy += round(learner.score(test_x, test_y) * 100, 5)
    # find arv prediction
    print(accumulated_accuracy / k)
    return accumulated_accuracy / k


print(data.head())

print("DT")
k_fold_cross_validation(5, DecisionTreeClassifier(), data)
print("RF")
k_fold_cross_validation(5, RandomForestClassifier(n_estimators=100), data)
print("AdaBoost")
k_fold_cross_validation(5, AdaBoostClassifier(n_estimators=200), data)
print("Gradient Boosting")
k_fold_cross_validation(5, GradientBoostingClassifier(n_estimators=200), data)
print("SVC")
k_fold_cross_validation(5, SVC(gamma='auto'), data)
print("KNN")
k_fold_cross_validation(5, KNeighborsClassifier(n_neighbors=3), data)
print("Naive Bayes")
k_fold_cross_validation(5, GaussianNB(), data)
print("SGD")
k_fold_cross_validation(5, SGDClassifier(), data)
print("Perceptron")
k_fold_cross_validation(5, Perceptron(), data)
