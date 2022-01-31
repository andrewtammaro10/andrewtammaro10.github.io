# andrewtammaro10.github.io

Write Up
Introduction
How much is an interception worth? My roommate and I recently had a debate over Trevon Diggs, the Dallas Cowboys All-Pro cornerback. While he led the National Football League in interceptions in the 2021-2022 season, his coverage grades were among some of the worst in the league. I was curious as to how many points a defender saves his team every time he picks off a pass from the opposing quarterback.

I set out to answer this question by first obtaining play-by-play data from Kaggle, containing 256 variables such as the quarter, yards to go, teams, play description, and many more. The data pertained to seasons from 2009 to 2016. The process consisted of data wrangling, modeling, predicting, and final aggregating before it could be uploaded to Tableau for visual analysis.


Data Wrangling
The original dataset contained 449,371 plays. The first step was to subset this down to the 2017 and 2018 seasons in order to construct the models. The 256 columns were also subset down to 32 that contained relevant pre-play information and could be used to model or aggregate. Additional columns for year and sum of home and away scores were also created at this point.

The next task was to create the target variable that the models would be trained on. The goal was to create a model that would estimate the outcome of the drive in terms of points. To achieve this, I looped over each drive of each game, pulling the score from the possession team on the last play of the drive. This score was then applied to each observation as ‘drive points’. An important distinction for me was to ensure that drives that ended in a defensive touchdown were recorded as 0 points for the possession team, as opposed to -6. The model is intended to estimate how many points the drive would be worth for the offensive team, which does not include the points a defense could score. The final step before the modeling phase was to subset the remaining plays down to just runs or passes, which are core offensive plays.


Modeling
With a concrete data set of plays to build models on, the correct variables had to be selected and put in a format that could be modeled on. One-hot encoding was used to prepare the categorical variables to be put into the model. Missing observations were dropped from the data set. The target variable, ‘drive points’, was set and separated from the input variables. In order to conduct model validation, the data was split into training and test datasets, reproducible due to a specified random state (#12 for the GOAT).

Two models were ultimately used. First, a linear regression was run on the data. The MAE (Mean Absolute Error) on the test data set was 2.204. Second, a random forest was fitted on the same training data converted to an array. 1,000 decision trees and the same random state were specified. After applying the model to the test data, the MAE was only 2.206. A variable importance plot indicated which inputs contributed most to predicting drive points. As seen below, distance from the goal line was the most important variable, followed by time remaining in the half, time remaining in the game, score differential, and the total score in the game. All of these intuitively make sense to football fans as factors that would most influence the outcome of a drive.

