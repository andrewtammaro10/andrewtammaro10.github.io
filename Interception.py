#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 18:03:48 2022

@author: drewtammaro
"""

### Modules ###
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


################################
#  Step 1: Posturing the Data  #
################################

# Load the data frame
pbp = pd.read_csv("/Users/drewtammaro/Downloads/NFL Play by Play 2009-2018 (v5).csv")

# Find the date values
pbp.game_date.unique()

# Pull off year
pbp['year'] = pbp['game_date'].str[:4].astype(int)
pbp.year.unique()

# Add total score column
pbp['score_sum'] = pbp['total_home_score'] + pbp['total_away_score']

# Subset for 2017 and 2018 seasons
pbp1718 = pbp[pbp["year"] >= 2017]

# Pull a full column list
list(pbp1718.columns)

# Keep specific columns relevant to the problem
pbp1718 = pbp1718[['game_id', 'posteam', 'home_team', 'away_team',
 'posteam_type', 'defteam', 'yardline_100', 'half_seconds_remaining',
 'game_seconds_remaining', 'game_half', 'drive', 'qtr', 'down',
 'goal_to_go', 'ydstogo', 'desc', 'play_type' 'td_team',
 'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
 'total_home_score', 'total_away_score', 'score_sum', 'posteam_score',
 'defteam_score', 'score_differential', 'interception', 'sack', 'touchdown',
 'passer_player_name', 'interception_player_name', 'year']]

# Reset the index
pbp1718 = pbp1718.reset_index(drop=True)


###################################################
#  Step 2: Creating Drive Points Target Variable  #
###################################################

# Create new columns for drive score loops
home_drive_points=pd.Series(np.zeros(len(pbp1718)))
away_drive_points=pd.Series(np.zeros(len(pbp1718)))
drive_points=pd.Series(np.zeros(len(pbp1718)))
scoring_drive=pd.Series(np.zeros(len(pbp1718)))


# Loop over each drive in each game
k=0
for game in pbp1718.game_id.unique():
    game_slice = pbp1718[pbp1718.game_id==game]
    for drive in game_slice.drive.unique():
        # Slice the dataframe to get just the data for the give game/drive
        game_drive = game_slice[game_slice.drive==drive]
        
        if len(game_drive)==0:
            continue
        
        # Determine who scored
        home_score_diff = game_drive.total_home_score.iloc[-1] - game_drive.total_home_score.iloc[0]
        away_score_diff = game_drive.total_away_score.iloc[-1] - game_drive.total_away_score.iloc[0]
        
        # Home Score
        if game_drive.posteam_type.iloc[0] == 'home':
            if home_score_diff>0:
                
                
                scoring_drive.loc[game_drive.index] = 1
                home_drive_points.loc[game_drive.index] = home_score_diff
                drive_points.loc[game_drive.index] = home_score_diff
                
        # Away Score   
        if game_drive.posteam_type.iloc[0] == 'away':
            if away_score_diff > 0:
                
                scoring_drive.loc[game_drive.index] = 1
                away_drive_points.loc[game_drive.index] = away_score_diff
                drive_points.loc[game_drive.index] = away_score_diff
        
        # Update the parent dataframe for this drive             
        k+=1
        
# Merge columns back into data frame
pbp1718['home_drive_points']=home_drive_points
pbp1718['away_drive_points']=away_drive_points
pbp1718['drive_points']=drive_points
pbp1718['scoring_drive']=scoring_drive


## Subset to only include pass and run plays (offensive plays)
pbp1718.play_type.unique()
# ['kickoff', 'pass', 'punt', 'run', 'extra_point', 'field_goal',
#       'no_play', nan, 'qb_spike', 'qb_kneel']
pbp1718 = pbp1718[pbp1718['play_type'].isin(['pass', 'run'])]



####################################
#  Step 3: Linear Regression Model #
####################################

## Model prep-processing
# Define categorical variables
pbp1718["posteam_type"] = pbp1718["posteam_type"].astype("category")
pbp1718["game_half"] = pbp1718["game_half"].astype("category")
pbp1718["qtr"] = pbp1718["qtr"].astype("category")
pbp1718["down"] = pbp1718["down"].astype("category")
pbp1718["goal_to_go"] = pbp1718["goal_to_go"].astype("category")
pbp1718["posteam_timeouts_remaining"] = pbp1718["posteam_timeouts_remaining"].astype("category")
pbp1718["defteam_timeouts_remaining"] = pbp1718["defteam_timeouts_remaining"].astype("category")

# Set model variables
X = pbp1718[['drive_points', 'posteam_type', 'yardline_100',
             'half_seconds_remaining', 'game_seconds_remaining', 
             'game_half', 'drive', 'qtr', 'down', 'goal_to_go', 
             'ydstogo', 'posteam_timeouts_remaining',
             'defteam_timeouts_remaining', 'score_sum', 'score_differential']]

# Drop NAs
X = X.dropna()

# One-hot encoding
X = pd.get_dummies(data=X, drop_first=True)
X.head()

## Redefine prepared dataframe so it can still be used for Random Forest
X1 = X

# Set target variable
Y1 = X1['drive_points']

# Drop the target variabe from the inputs
X1 = X1.drop('drive_points', axis = 1)

# Train/test split (70/30)
x1_train, x1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size = 0.30, random_state = 12)

# Check the shape
print('x1_train Shape:', x1_train.shape)
print('y1_train Shape:', y1_train.shape)
print('x1_test Shape:', x1_test.shape)
print('y1_test Shape:', y1_test.shape)

# Create the model object
lm = LinearRegression()

# Train the model using the training data
lm.fit(x1_train, y1_train)

# Print coefficients
coeff1_parameter = pd.DataFrame(lm.coef_,X1.columns,columns=['Coefficient'])
coeff1_parameter

# Predict on the test data
predictions = lm.predict(x1_test)

# Calculate the absolute errors
errors = abs(predictions - y1_test)

# Print out the mean absolute error (MAE)
print('Mean Absolute Error:', round(np.mean(errors), 3), 'degrees.')
# 2.204 degrees




################################
#  Step 4: Random Forest Model #
################################

## Random forest model
# Set target variable in array
Y2 = np.array(X['drive_points'])

# Drop the target variabe from the inputs
X2 = X.drop('drive_points', axis = 1)

# Saving feature names for later use
feature_list = list(X2.columns)

# Convert to numpy array
X2 = np.array(X2)

# Train/test split (70/30)
x2_train, x2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size = 0.30, random_state = 12)

# Check the shape
print('x2_train Shape:', x2_train.shape)
print('y2_train Shape:', y2_train.shape)
print('x2_test Shape:', x2_test.shape)
print('y2_test Shape:', y2_test.shape)

# Import the Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 12)

# Train the model using the training data
rf.fit(x2_train, y2_train);

# Predict on the test data
predictions = rf.predict(x2_test)

# Calculate the absolute errors
errors = abs(predictions - y2_test)

# Print out the mean absolute error (MAE)
print('Mean Absolute Error:', round(np.mean(errors), 3), 'degrees.')
# 2.026 degrees




#####################################
#  Step 5: Variable Importance Plot #
#####################################

# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# Set the plot style
plt.style.use('fivethirtyeight')

# List of x locations for plotting
x_values = list(range(len(importances)))

# Make the bar chart
plt.bar(x_values, importances, orientation = 'vertical');plt.xticks(x_values, feature_list, rotation='vertical');plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importance');




##################################
#  Step 6: Applying Predictions  #
##################################


# Subset data for 2009-2016
pbp0916 = pbp[pbp["year"] < 2017]

# Subset the columns down
pbp0916 = pbp0916[['game_id', 'posteam', 'posteam_type', 'defteam',
 'yardline_100', 'half_seconds_remaining', 'game_seconds_remaining',
 'game_half', 'drive', 'qtr', 'down', 'goal_to_go', 'ydstogo', 'desc',
 'play_type', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
 'total_home_score', 'total_away_score', 'score_sum', 'posteam_score',
 'defteam_score', 'score_differential', 'interception', 'sack', 'touchdown',
 'passer_player_name', 'interception_player_name', 'year']]

# Subset for pass and run plays
pbp0916 = pbp0916[pbp0916['play_type'].isin(['pass', 'run'])]

# Split to remove non-input variables we need later
pbpdrop = pbp0916[['posteam', 'defteam','year', 'passer_player_name',
                   'interception_player_name', 'desc', 'interception',
                   'touchdown']]
pbp0916 = pbp0916.drop(['defteam','year', 'passer_player_name',
                   'interception_player_name', 'desc'], axis = 1)

# Define categorical variables
pbp0916["posteam_type"] = pbp0916["posteam_type"].astype("category")
pbp0916["game_half"] = pbp0916["game_half"].astype("category")
pbp0916["qtr"] = pbp0916["qtr"].astype("category")
pbp0916["down"] = pbp0916["down"].astype("category")
pbp0916["goal_to_go"] = pbp0916["goal_to_go"].astype("category")
pbp0916["posteam_timeouts_remaining"] = pbp0916["posteam_timeouts_remaining"].astype("category")
pbp0916["defteam_timeouts_remaining"] = pbp0916["defteam_timeouts_remaining"].astype("category")

# Set specific input variables
pbp0916 = pbp0916[['posteam_type', 'yardline_100',
             'half_seconds_remaining', 'game_seconds_remaining', 
             'game_half', 'drive', 'qtr', 'down', 'goal_to_go', 
             'ydstogo', 'posteam_timeouts_remaining',
             'defteam_timeouts_remaining', 'score_sum', 'score_differential']]


# One-hot encoding
pbp0916 = pd.get_dummies(data= pbp0916, drop_first=True)

# Get predictions
pbp0916['drive_pred'] = rf.predict(pbp0916)

# Concatenate other variables back on
pbp0916 = pd.concat([pbp0916, pbpdrop], axis=1)

# Subset for interceptions
pbppicks = pbp0916[pbp0916['interception'] == 1]

# Create points saved column
pbppicks['points_saved'] = np.where(pbppicks['touchdown'] == 0, pbppicks['drive_pred'], (pbppicks['drive_pred'] + 6.979))
 
# Subset for final output columns
pbpfinal = pbppicks[['posteam', 'defteam','year', 'passer_player_name',
                   'interception_player_name', 'desc', 'points_saved']]

# Replace duplicate teams
pbpfinal.posteam.unique()
pbpfinal["posteam"].replace({"JAC": "JAX", "LA": "STL"}, inplace=True)
pbpfinal["defteam"].replace({"JAC": "JAX", "LA": "STL"}, inplace=True)

# Set directory
os.chdir('/Users/drewtammaro/desktop/IAA/Python')

# Write to csv for individual stat Tableau work
pbpfinal.to_csv('pbpfinal.csv')

# Split and aggregate by possession team and year
pbpoffense = pbpfinal[['posteam', 'year', 'points_saved']]
pbpoffense = pbpoffense.rename(columns={'posteam': 'team','points_saved': 'points_lost'})
pbpoffense = pbpoffense.groupby(['team', 'year'])['points_lost'].agg('sum').reset_index()

# Split and aggregate by defensive team and year
pbpdefense = pbpfinal[['defteam', 'year', 'points_saved']]
pbpdefense = pbpdefense.rename(columns={'defteam': 'team'})
pbpdefense = pbpdefense.groupby(['team', 'year'])['points_saved'].agg('sum').reset_index()

# Merge together possession and defensive team data
pbpagg = pd.merge(pbpoffense, pbpdefense, on=['team','year'])

# Write to csv for team stat Tableau work
pbpagg.to_csv('pbpagg.csv')



        
