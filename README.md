# Football Match Predictor


## Table of Contents  

- [Introduction](#introduction)
- [Data & Features](#data-&-features)
- [Methods](#methods)
- [Demo](#demo)
- [Installation](#installation)
- [File Structure](#file-structure)
- [Architecture ](#architecture)

## Introduction
Welcome to our revolutionary football prediction project! Our goal is simple yet revolutionary: to reshape how soccer fans interact with the sport.
 By harnessing the power of machine learning, historical match data, and advanced web tech, we're empowering users to not only make informed choices
 for upcoming matches but also to refine their betting strategies. In today's digital landscape, researching matches can be overwhelming—our solution?
 A user-friendly web app merging machine learning and intuitive design. Here, users access predictions, live scores, and more, all seamlessly
 integrated. Say goodbye to exhaustive research and hello to data-driven predictions, elevating your soccer experience like never before.

## Data & Features

Our football prediction project draws its strength from an expansive dataset encompassing five premier leagues: 
- ![English Premier League](https://raw.githubusercontent.com/stevenrskelton/flag-icon/master/png/16/country-4x3/gb.png "English Premier League") English Premier League
- ![ Italian Serie A](https://raw.githubusercontent.com/stevenrskelton/flag-icon/master/png/16/country-4x3/it.png " Italian Serie A")  Italian Serie A
- ![German Bundesliga](https://raw.githubusercontent.com/stevenrskelton/flag-icon/master/png/16/country-4x3/de.png "German Bundesliga") German Bundesliga
- ![ Spanish La Liga](https://raw.githubusercontent.com/stevenrskelton/flag-icon/master/png/16/country-4x3/es.png " Spanish La Liga")  Spanish La Liga
- ![French Ligue 1](https://raw.githubusercontent.com/stevenrskelton/flag-icon/master/png/16/country-4x3/fr.png "French Ligue 1") French Ligue 1

The dataset spans an impressive timeline from the 2009-2010 season to the current 2022-2023 season, ensuring a comprehensive analysis of 
trends and patterns over the years. Organized meticulously, each season's data is stored within a dedicated CSV file.

Within this treasure trove of information, our focus is on a curated selection of features that have been demonstrated to be highly relevant in
 predicting match outcomes. These features are:

- **Team:** Identifying the participating teams in a match, a fundamental piece of information.
- **HS (Home Shots):** The overall shots taken by the home team.
- **AS (Away Shots):** The overall shots taken by the away team.
- **HST (Home Shots On Target):** The overall shots on target taken by the home team.
- **AST (Away Shots On Target):** The overall shots on target taken by the away team.

The features above are the features provided by the dataset. This features are not enough to get good and reliable results. Because of that we added 4 more features to our machine learning algorithms:
- **HAS (Home Attack Strength):** Average goals scored by the home team divided by the average goals scored by any home team.
- **HDS (Home Defence Strength):** Average goals conceded by the home team divided by the average goals conceded by any home team.
- **AAS (Away Attack Strength):** Average goals scored by the away team divided by the average goals scored by any away team.
- **ADS (Away Defence Strength):** Average goals conceded by the away team divided by the average goals conceded by any away team.
- **pastGoalDiff (Past Games Goal Difference):** The goal differenc in the last 3 games of a team.


These meticulously chosen features collectively represent the core dynamics of a football match, covering attacking and defensive strengths and
 tendencies for both home and away teams. By meticulously processing and analyzing these factors, our prediction model gains a deep understanding 
of each team's strengths and weaknesses, allowing for accurate forecasts of match outcomes.

## Methods
