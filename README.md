# Football Match Predictor ⚽


## Table of Contents  

- [Introduction](#introduction)
- [Data & Features](#data--features)
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

As we set out to build our football match prediction system, we delved into the world of machine learning techniques and strategies. By analyzing past football results, we trained various models using the `Predictions.py` to predict future match outcomes accurately. This allowed us to improve the reliability of our predictions, making match forecasting as easy as supporting your favorite team.

- We explored various machine learning models – such as:
  - **Support Vector Classification (SVC):** It's like a pattern finder, helping us predict matches, especially when they get tricky. SVC can see hidden patterns, making it useful for complicated matches.
  -  **Logistic Regression:** This one is like a smart detective, noticing simple trends in match data. It's great at figuring out whether a team might win or not.
  -  **Multi-Layer Perceptrons (MLP):** Imagine a puzzle solver – MLP is really good at finding hidden patterns in data. This helps us understand why matches end the way they do.
  -  **Random Forest:** Think of a team brainstorming ideas – that's how Random Forest works. It gathers ideas from different models, making it strong at predicting match results.
  -  **K-Nearest Neighbors (KNN):** This one is like asking your friends for advice. KNN checks similar matches to help predict the outcome of a match.

By looking at historical football results, we trained these models to learn from the past and predict the future. It's like learning from your mistakes to do better next time. We also listened to feedback from users and made our models better over time, like a team that practices and improves.

After a close look at all the options, we made our system strong, predicting matches with high accuracy and tested the past matches to see how well they predicted real match outcomes. Since matches can end in three ways – *Home win __(H)__, Away win __(A)__, or a Draw __(D)__* – we considered all these possibilities to make our predictions even better.


## Demo

| Home page | Matches page | Login page  |
| --- | --- | ---  |
|  | | 

| Blog page | Prediction page | 
| --- | --- | 
|  |   | 



Step into our web app world, where soccer excitement meets smart technology. We've built a cool website using JavaScript and a powerful Django server. It's like your ultimate soccer sidekick, offering features that fans love:

- **Game Predictions**: Access to accurate and reliable match predictions based on historical data and machine learning algorithms.
- **Team Information** : Detailed information on team performance, and historical match results.
- **Live Score Updates**: Real-time updates on ongoing matches, ensuring users stay up to date with the latest scores.
- **User-Friendly Interface**: A responsive and visually appealing interface that provides seamless navigation and easy access to predictions and other features (HTML/JS/Python).
- Fast Performance: Optimized loading times for games, ensuring a smooth and reliable user experience


## Installation

To use our tool football Predition, follow these installation steps:

1. Clone the repository:

   ```bash
    git clone https://github.com/lielZilberman/Football-Prediction-Project.git

2. Install all the required modules
    ```bash
   npm install

3. Enter into the `server` directory and run:
   ```bash
   python3 manage.py runserver

## File Structure

Within the download you'll find the following directories and files:

```
FOOTBALL-PREDICTION-PROJECT
  ├── data
  ├── node_modules
  ├── server
  |     ├── server
  |     ├── manage.py
  |     └── web
  |         ├── staric
  |         ├── server
  |         ├── urls.py
  |         └── views.py
  ├── Predictions.py
  └── package-lock.json
```



