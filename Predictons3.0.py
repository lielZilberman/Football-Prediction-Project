import keras as keras
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from IPython.display import display
from os import path, makedirs, walk
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
import numpy as np

en_data_folder = 'english-premier-league_zip'
es_data_folder = 'spanish-la-liga_zip'
fr_data_folder = 'french-ligue-1_zip'
ge_data_folder = 'german-bundesliga_zip'
it_data_folder = 'italian-serie-a_zip'

data_folders = [en_data_folder, es_data_folder,
                fr_data_folder, ge_data_folder, it_data_folder]
season_range = (17, 22)
en_data = []
es_data = []
fr_data = []
ge_data = []
it_data = []


def org_data(data, data_folder):
    for season in range(season_range[0], season_range[1] + 1):
        data.append(
            'data/{}/data/season-{:02d}{:02d}_csv.csv'.format(data_folder, season, season + 1))
    data_frame = []
    for data_file in data:
        if path.exists(data_file):
            data_frame.append(pd.read_csv(data_file))
    return pd.concat(data_frame, ignore_index=True)


en_merge_data = org_data(en_data, en_data_folder)
es_merge_data = org_data(es_data, es_data_folder)
ge_merge_data = org_data(ge_data, ge_data_folder)
fr_merge_data = org_data(fr_data, fr_data_folder)
it_merge_data = org_data(it_data, it_data_folder)

res_16 = en_merge_data.iloc[:, :23]
res_16 = res_16.drop(['Div', 'Date'], axis=1)
# table_features = en_merge_data.iloc[:, :7]
# print(table_features)
# print("***********************")
# table_features = table_features.drop(['FTHG', 'FTAG', 'Div', 'Date'], axis=1)
feature_table = en_merge_data.iloc[:, :23]

# Team, Home Goals Score, Away Goals Score, Attack Strength (lines 75, 76), Home Goals Conceded, Away Goals Conceded, Defensive Strength (lines 77, 78)
table_16 = pd.DataFrame(columns=('Team', 'HGS', 'AGS', 'HAS', 'AAS', 'HGC', 'AGC', 'HDS', 'ADS',))
table_16 = table_16[:-1]
res_16 = res_16[:-1]

avg_home_scored_16 = res_16.FTHG.sum() * 1.0 / res_16.shape[0]
avg_away_scored_16 = res_16.FTAG.sum() * 1.0 / res_16.shape[0]
avg_home_conceded_16 = avg_away_scored_16
avg_away_conceded_16 = avg_home_scored_16
res_home = res_16.groupby('HomeTeam')
res_away = res_16.groupby('AwayTeam')
t = res_home.HomeTeam.apply(pd.DataFrame)
table_16.Team = t.columns.values
table_16.HGS = res_home.FTHG.sum().values
table_16.HGC = res_home.FTAG.sum().values
table_16.AGS = res_away.FTAG.sum().values
table_16.AGC = res_away.FTHG.sum().values
num_games = res_16.shape[0] / 20
table_16.HAS = (table_16.HGS / num_games) / avg_home_scored_16
table_16.AAS = (table_16.AGS / num_games) / avg_away_scored_16
table_16.HDS = (table_16.HGC / num_games) / avg_home_conceded_16
table_16.ADS = (table_16.AGC / num_games) / avg_away_conceded_16

feature_table = feature_table[['HomeTeam', 'AwayTeam', 'FTR', 'HST', 'AST']]
f_HAS = []
f_HDS = []
f_AAS = []
f_ADS = []
for index, row in feature_table.iterrows():
    f_HAS.append(table_16[table_16['Team'] == row['HomeTeam']]['HAS'].values[0])
    f_HDS.append(table_16[table_16['Team'] == row['HomeTeam']]['HDS'].values[0])
    f_AAS.append(table_16[table_16['Team'] == row['AwayTeam']]['AAS'].values[0])
    f_ADS.append(table_16[table_16['Team'] == row['AwayTeam']]['ADS'].values[0])

feature_table['HAS'] = f_HAS
feature_table['HDS'] = f_HDS
feature_table['AAS'] = f_AAS
feature_table['ADS'] = f_ADS


def transformResult(row):
    '''Converts results (H,A or D) into numeric values'''
    if row.FTR == 'H':
        return 1
    elif row.FTR == 'A':
        return 2
    else:
        return 0


feature_table["Result"] = feature_table.apply(lambda row: transformResult(row), axis=1)

feature_table = feature_table[:-1]  # All but last game week
feat_table = res_16.sort_index(ascending=False)
feat_table = feat_table[['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']]
# Adding next week fixtures
new_fixtures = pd.DataFrame([['Southampton', 'Liverpool', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=feat_table.columns)
# new_feat_table = new_fixtures.concat(feat_table)
# new_feat_table = pd.concat([feat_table, new_fixtures], ignore_index=True)
# new_feat_table = new_feat_table.sort_index(ascending=False)
# new_feat_table = new_feat_table.reset_index().drop(['index'], axis=1)
# new_feat_table = new_feat_table.sort_index(ascending=False)
new_feat_table = pd.concat([new_fixtures, feat_table], ignore_index=True)
new_feat_table = new_feat_table.sort_index(ascending=False)
new_feat_table = new_feat_table.reset_index().drop(['index'], axis=1)
new_feat_table = new_feat_table.sort_index(ascending=False)
feat_table = new_feat_table
# print(feat_table)
# print("feat table 1: ", feat_table)
# print("*****************")
feat_table["pastHS"] = 0.0
feat_table["pastHC"] = 0.0
feat_table["pastAS"] = 0.0
feat_table["pastAC"] = 0.0
feat_table["pastHG"] = 0.0
feat_table["pastAG"] = 0.0

k = 3
for i in range(feat_table.shape[0] - 1, -1, -1):
    row = feat_table.loc[i]
    ht = row.HomeTeam
    at = row.AwayTeam
    ht_stats = feat_table[(feat_table.HomeTeam == ht) | (feat_table.AwayTeam == ht)].loc[i - 1:-1].head(k)
    # ht_stats = feat_table.loc[i - 1:-1][(feat_table.HomeTeam == ht) | (feat_table.AwayTeam == ht)].head(k)
    # at_stats = feat_table.loc[i - 1:-1][(feat_table.HomeTeam == at) | (feat_table.AwayTeam == at)].head(k)
    at_stats = feat_table[(feat_table.HomeTeam == at) | (feat_table.AwayTeam == at)].loc[i - 1:-1].head(k)

    feat_table.at[i, 'pastHC'] = (ht_stats[ht_stats["AwayTeam"] == ht].sum().HC + ht_stats[
        ht_stats["HomeTeam"] == ht].sum().HC) / k
    feat_table.at[i, 'pastAC'] = (at_stats[at_stats["AwayTeam"] == at].sum().HC + at_stats[
        at_stats["HomeTeam"] == at].sum().HC) / k
    feat_table.at[i, 'pastHS'] = (ht_stats[ht_stats["AwayTeam"] == ht].sum().HS + ht_stats[
        ht_stats["HomeTeam"] == ht].sum().AS) / k
    feat_table.at[i, 'pastAS'] = (at_stats[at_stats["AwayTeam"] == at].sum().HS + at_stats[
        at_stats["HomeTeam"] == at].sum().AS) / k
    feat_table.at[i, 'pastHG'] = (ht_stats[ht_stats["AwayTeam"] == ht].sum().FTAG + ht_stats[
        ht_stats["HomeTeam"] == ht].sum().FTHG) / k
    feat_table.at[i, 'pastAG'] = (at_stats[at_stats["AwayTeam"] == at].sum().FTAG + at_stats[
        at_stats["HomeTeam"] == at].sum().FTHG) / k

f_HAS = []
f_HDS = []
f_AAS = []
f_ADS = []
for index, row in feat_table.iterrows():
    # print row
    f_HAS.append(table_16[table_16['Team'] == row['HomeTeam']]['HAS'].values[0])
    f_HDS.append(table_16[table_16['Team'] == row['HomeTeam']]['HDS'].values[0])
    f_AAS.append(table_16[table_16['Team'] == row['HomeTeam']]['AAS'].values[0])
    f_ADS.append(table_16[table_16['Team'] == row['HomeTeam']]['ADS'].values[0])
feat_table['HAS'] = f_HAS
feat_table['HDS'] = f_HDS
feat_table['AAS'] = f_AAS
feat_table['ADS'] = f_ADS

# feat_table = feat_table.drop(['FTHG', 'FTAG', 'HS', 'AS', 'HC', 'AC'], axis=1)
feat_table = feat_table.drop(['FTHG', 'FTAG', 'HC', 'AC'], axis=1)

feat_table["Result"] = feat_table.apply(lambda row: transformResult(row), axis=1)
feat_table.sort_index(inplace=True)
feat_table["pastCornerDiff"] = (feat_table["pastHC"] - feat_table["pastAC"]) / k
feat_table["pastGoalDiff"] = (feat_table["pastHG"] - feat_table["pastAG"]) / k
feat_table["pastShotsDiff"] = (feat_table["pastHS"] - feat_table["pastAG"]) / k
feat_table = feat_table.fillna(0)
num_games = feat_table.shape[0] - 1
# print(num_games)
# print(feature_table)
v_split = 450
n_games = num_games - v_split
feat_table.drop(['pastHC', 'pastAS', 'pastAC', 'pastHG', 'pastAG'], axis=1)
# print(feature_table)
temp_table = feat_table[:-1]
X = temp_table[['HS', 'AS', 'HST', 'AST', 'pastGoalDiff', 'HAS', 'HDS', 'AAS', 'ADS']]
Y = temp_table['Result']
# X_train = feat_table.loc[0:n_games, ['HAS', 'HDS', 'AAS', 'ADS']]
# y_train = feat_table.loc[0:n_games, ['Result']]
# X_test = feat_table.loc[n_games:num_games, ['HAS', 'HDS', 'AAS', 'ADS']]
# y_test = feat_table.loc[n_games:num_games, ['Result']]
# X_predict = feat_table.loc[num_games:, ['pastCornerDiff', 'pastGoalDiff', 'pastShotsDiff', 'HAS', 'HDS', 'AAS', 'ADS']]
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

X_train = feat_table.loc[0:n_games, ['HS', 'AS', 'HST', 'AST', 'pastGoalDiff', 'HAS', 'HDS', 'AAS', 'ADS']]
y_train = feat_table.loc[0:n_games, ['Result']]
X_test = feat_table.loc[n_games:num_games - 1, ['HS', 'AS', 'HST', 'AST', 'pastGoalDiff', 'HAS', 'HDS', 'AAS', 'ADS']]
y_test = feat_table.loc[n_games:num_games - 1, ['Result']]
X_predict = feat_table[['HS', 'AS', 'HST', 'AST', 'pastGoalDiff', 'HAS', 'HDS', 'AAS', 'ADS']].loc[num_games:]
rfm = RandomForestClassifier(n_estimators=100, random_state=42)
rfm.fit(X_train, np.ravel(y_train))
res = rfm.predict(X_test)
print(accuracy_score(y_test, res))
# # KNN
clf_knn = KNeighborsClassifier(n_neighbors=48)
clf_knn.fit(X_train, np.ravel(y_train))
scores = accuracy_score(y_test, clf_knn.predict(X_test))
print(scores)
# plot_scores_knn = []
# for b in range(1, 50):
#     clf_knn = KNeighborsClassifier(n_neighbors=b)
#     clf_knn.fit(X_train, np.ravel(y_train))
#     scores = accuracy_score(y_test, clf_knn.predict(X_test))
#     plot_scores_knn.append(scores)
plot_scores_XGB = []
for i in range(1, 100):
    clf_XGB = XGBClassifier(n_estimators=i, max_depth=100)
    clf_XGB.fit(X_train, np.ravel(y_train))
    scores = accuracy_score(y_test, clf_XGB.predict(X_test))
    plot_scores_XGB.append(scores)

# max_knn_n = max(plot_scores_knn)
# max_knn_ind = plot_scores_knn.index(max_knn_n)
#
max_XGB_e = max(plot_scores_XGB)
max_XGB_ind = plot_scores_XGB.index(max_XGB_e) if plot_scores_XGB.index(max_XGB_e) != 0 else 1


# print(max_knn_n, max_knn_ind)
print(max_XGB_e, max_XGB_ind)
# print(max_logreg_c, max_logreg_ind)
#
# clf_knn = KNeighborsClassifier(n_neighbors=max_knn_ind).fit(X_train, np.ravel(y_train))
# clf_XGB = XGBClassifier(n_estimators=max_XGB_ind).fit(X_train, y_train)

y_pred_knn = clf_knn.predict(X_predict)
# y_pred_rfm = rfm.predict(X_predict)

this_week = feat_table[['HomeTeam', 'AwayTeam']][num_games:]
this_week['Result_knn'] = y_pred_knn

# this_week['Result_rfm'] = y_pred_rfm


def transformResultBack(row, col_name):
    if row[col_name] == 1:
        return 'H'
    elif row[col_name] == 2:
        return 'A'
    else:
        return 'D'


this_week["Res_knn"] = this_week.apply(lambda row: transformResultBack(row, "Result_knn"), axis=1)
# this_week["Res_rfm"] = this_week.apply(lambda row: transformResultBack(row, "Result_rfm"), axis=1)
# this_week["Res_logreg"] = this_week.apply(lambda row: transformResultBack(row, "Result_logreg"), axis=1)

# this_week.drop(["Result_knn", "Result_rfm"], axis=1, inplace=True)
this_week.drop(["Result_knn"], axis=1, inplace=True)
print(this_week)
