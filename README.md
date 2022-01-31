# NFL Interception Value
## Andrew Tammaro

### Introduction
How much is an interception worth? My roommate and I recently had a debate over Trevon Diggs, the Dallas Cowboys All-Pro cornerback. While he led the National Football League in interceptions in the 2021-2022 season, his coverage grades were among some of the worst in the league. I was curious as to how many points a defender saves his team every time he picks off a pass from the opposing quarterback.

I set out to answer this question by first obtaining play-by-play data from Kaggle, containing 256 variables such as the quarter, yards to go, teams, play description, and many more. The data pertained to seasons from 2009 to 2016. The process consisted of data wrangling, modeling, predicting, and final aggregating before it could be uploaded to Tableau for visual analysis.


### Data Wrangling
The original dataset contained 449,371 plays. The first step was to subset this down to the 2017 and 2018 seasons in order to construct the models. The 256 columns were also subset down to 32 that contained relevant pre-play information and could be used to model or aggregate. Additional columns for year and sum of home and away scores were also created at this point.

The next task was to create the target variable that the models would be trained on. The goal was to create a model that would estimate the outcome of the drive in terms of points. To achieve this, I looped over each drive of each game, pulling the score from the possession team on the last play of the drive. This score was then applied to each observation as ‘drive points’. An important distinction for me was to ensure that drives that ended in a defensive touchdown were recorded as 0 points for the possession team, as opposed to -6. The model is intended to estimate how many points the drive would be worth for the offensive team, which does not include the points a defense could score. The final step before the modeling phase was to subset the remaining plays down to just runs or passes, which are core offensive plays.


### Modeling
With a concrete data set of plays to build models on, the correct variables had to be selected and put in a format that could be modeled on. One-hot encoding was used to prepare the categorical variables to be put into the model. Missing observations were dropped from the data set. The target variable, ‘drive points’, was set and separated from the input variables. In order to conduct model validation, the data was split into training and test datasets, reproducible due to a specified random state (#12 for the GOAT).

Two models were ultimately used. First, a linear regression was run on the data. The MAE (Mean Absolute Error) on the test data set was 2.204. Second, a random forest was fitted on the same training data converted to an array. 1,000 decision trees and the same random state were specified. After applying the model to the test data, the MAE was only 2.206. A variable importance plot indicated which inputs contributed most to predicting drive points. As seen below, distance from the goal line was the most important variable, followed by time remaining in the half, time remaining in the game, score differential, and the total score in the game. All of these intuitively make sense to football fans as factors that would most influence the outcome of a drive.

<p align="center">
  <img src="https://user-images.githubusercontent.com/86579251/151810782-4cc478ec-1a25-4ab9-8e7c-8098528b37a7.png" alt="Variable Importance Plot"/>
</p>

With the final model selected, the only thing left to do before visualizing the data was to apply the model to the play-by-play data from 2009 to 2016. To do this, the same steps from before were followed to prepare the data. Additionally, on plays where the interception resulted in a defensive touchdown, 6.979 points (97.9% was the average extra point conversion rate) were added to the variable. Once applied, the data was subset down to only include plays on which an interception occurred.

I had two visualizations in mind, one for individual players and one for teams. For the individual player data, I simply needed to append the player names and year back onto the dataset. However, in order to compare how well a team fared on both sides of the ball, additional steps were required. For defensive teams, the predicted column was saved as ‘points saved’. However, for offensive teams, the predicted column was saved as ‘points lost’. This data was aggregated by team and year in order to produce the required data for the visualization. These two datasets were finalized and converted to a format that would be compatible with Tableau.


### Visualizing
In Tableau, I created three separate visuals. The first two show which players saved (defensive players) and lost (offensive players) the most points for their team in a given season. The top 10 players can be seen in the two graphs at the top of the dashboard. The third visual shows a comparison of how many points a team gained and lost in a specific season. Points saved is on the X-axis and points lost is on the Y-axis, indicating teams in the lower right corner as the most effective in terms of interceptions on both sides of the ball.


<iframe seamless frameborder="0" src="https://public.tableau.com/views/SideProject-Interceptions/Dashboard1?:embed=yes&:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link" width = '100%' height = '600'></iframe>


Over the span of the eight seasons in the data, Malcolm Jenkins (104.79) and Eli Manning (400.1) saved and lost their teams the most estimated drive points, respectively. Stevie Brown had the most prolific season on the defensive side, recording 8 interceptions for the Giants in 2012, saving them an estimated 33.88 points. Conversely, Drew Brees threw 19 interceptions and lost his team an estimated 78.34 points. Interestingly enough, Peyton Manning led the NFL in points lost with 55.56 in 2015, his last season. He accomplished this despite starting only 9 games and still managed to win his second Super Bowl.

On the team side, there were several interesting results seen in the scatterplot. The Packers and Jaguars sit in opposite corners when looking at the statistics over the full 8 seasons. Green Bay won 68.4% of their games in this span and also took home a Super Bowl, appearing in the playoffs in each season. On the other hand, the Jaguars won only 28.9% of their games and failed to make the playoffs once. The team that led the league in points saved won the Super Bowl twice in this eight year span, with the Saints in 2009 and the Seahawks in 2013. Interestingly enough, the team that led the league in points saved also took home the Lombardi Trophy twice in this span, with the Seahawks in 2013 (again) and the Patriots in 2016.


### Conclusion
This project, which started as a fun discussion, proved to be an incredibly fun and useful experience for me. In addition to helping me develop my data wrangling, modeling, Python, and Tableau skills, I saw firsthand the power of visualizations. Feel free to ask me any questions about the process or use this page to settle any sports debates with your roommates. Thanks for reading!

