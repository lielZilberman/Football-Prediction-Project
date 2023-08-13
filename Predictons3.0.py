import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from os import path, makedirs, walk
from sklearn.preprocessing import scale, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.neural_network import MLPClassifier

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


def compute(data):
    res_16 = data.iloc[:, :28]
    res_16 = res_16.drop(['Div', 'Date'], axis=1)
    feature_table = data.iloc[:, :28]

    table_16 = pd.DataFrame(columns=('Team', 'HGS', 'AGS', 'HAS', 'AAS', 'HGC', 'AGC', 'HDS', 'ADS'))
    # table_16 = table_16[:-1]
    # res_16 = res_16[:-1]

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
    return res_16, table_16, feature_table


en_res16, en_table16, en_feature_table = compute(en_merge_data)
es_res16, es_table16, es_feature_table = compute(es_merge_data)
ge_res16, ge_table16, ge_feature_table = compute(ge_merge_data)
fr_res16, fr_table16, fr_feature_table = compute(fr_merge_data)
it_res16, it_table16, it_feature_table = compute(it_merge_data)


def add_features(table, table16):
    f_HAS = []
    f_HDS = []
    f_AAS = []
    f_ADS = []
    for index, row in table.iterrows():
        f_HAS.append(table16[table16['Team'] == row['HomeTeam']]['HAS'].values[0])
        f_HDS.append(table16[table16['Team'] == row['HomeTeam']]['HDS'].values[0])
        f_AAS.append(table16[table16['Team'] == row['AwayTeam']]['AAS'].values[0])
        f_ADS.append(table16[table16['Team'] == row['AwayTeam']]['ADS'].values[0])

    table['HAS'] = f_HAS
    table['HDS'] = f_HDS
    table['AAS'] = f_AAS
    table['ADS'] = f_ADS
    return table


en_feature_table = add_features(en_feature_table, en_table16)
es_feature_table = add_features(es_feature_table, es_table16)
gr_feature_table = add_features(ge_feature_table, ge_table16)
fr_feature_table = add_features(fr_feature_table, fr_table16)
it_feature_table = add_features(it_feature_table, it_table16)


def transformResult(row):
    '''Converts results (H,A or D) into numeric values'''
    if row.FTR == 'H':
        return 1
    elif row.FTR == 'A':
        return 2
    else:
        return 0


en_feature_table["Result"] = en_feature_table.apply(lambda row: transformResult(row), axis=1)
es_feature_table["Result"] = es_feature_table.apply(lambda row: transformResult(row), axis=1)
ge_feature_table["Result"] = ge_feature_table.apply(lambda row: transformResult(row), axis=1)
fr_feature_table["Result"] = fr_feature_table.apply(lambda row: transformResult(row), axis=1)
it_feature_table["Result"] = it_feature_table.apply(lambda row: transformResult(row), axis=1)

en_final = en_res16.sort_index(ascending=False)
es_final = es_res16.sort_index(ascending=False)
ge_final = ge_res16.sort_index(ascending=False)
fr_final = fr_res16.sort_index(ascending=False)
it_final = it_res16.sort_index(ascending=False)

en_final = en_final[['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']]
es_final = es_final[['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']]
ge_final = ge_final[['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']]
fr_final = fr_final[['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']]
it_final = it_final[['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']]
en_new_fixtures = pd.DataFrame([['Southampton', 'Liverpool', 'D', 0, 0, 0, 0, 0, 0, 0, 0],
                             ['Man United', 'Fulham', 'D', 0, 0, 0, 0, 0, 0, 0, 0],
                             ['Leicester', 'West Ham', 'D', 0, 0, 0, 0, 0, 0, 0, 0],
                             ['Leeds', 'Tottenham', 'D', 0, 0, 0, 0, 0, 0, 0, 0],
                             ['Everton', 'Bournemouth', 'D', 0, 0, 0, 0, 0, 0, 0, 0],
                             ['Crystal Palace', 'Nott Forest', 'D', 0, 0, 0, 0, 0, 0, 0, 0],
                             ['Chelsea', 'Newcastle', 'D', 0, 0, 0, 0, 0, 0, 0, 0],
                             ['Brentford', 'Man City', 'D', 0, 0, 0, 0, 0, 0, 0, 0],
                             ['Aston Villa', 'Brighton', 'D', 0, 0, 0, 0, 0, 0, 0, 0],
                             ['Arsenal', 'Wolves', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=en_final.columns)


# Adding next week fixtures
# en_new_fixtures = pd.DataFrame([['Chelsea', 'Liverpool', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=en_final.columns)
es_new_fixtures = pd.DataFrame([['Valladolid', 'Getafe', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=es_final.columns)
ge_new_fixtures = pd.DataFrame([['Wolfsburg', 'Hertha', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=ge_final.columns)
fr_new_fixtures = pd.DataFrame([['Troyes', 'Lille', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=fr_final.columns)
it_new_fixtures = pd.DataFrame([['Udinese', 'Juventus', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=it_final.columns)


def new_table(fixtures, feat_table):
    new_feat_table = pd.concat([fixtures, feat_table], ignore_index=True)
    new_feat_table = new_feat_table.sort_index(ascending=False)
    new_feat_table = new_feat_table.reset_index().drop(['index'], axis=1)
    new_feat_table = new_feat_table.sort_index(ascending=False)
    feat_table = new_feat_table
    return feat_table


en_final = new_table(en_new_fixtures, en_final)
es_final = new_table(es_new_fixtures, es_final)
ge_final = new_table(ge_new_fixtures, ge_final)
fr_final = new_table(fr_new_fixtures, fr_final)
it_final = new_table(it_new_fixtures, it_final)


def handle(feat_table, table_16):
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

    feat_table = feat_table.drop(['FTHG', 'FTAG', 'HC', 'AC'], axis=1)

    feat_table["Result"] = feat_table.apply(lambda row: transformResult(row), axis=1)
    feat_table.sort_index(inplace=True)
    feat_table["pastCornerDiff"] = (feat_table["pastHC"] - feat_table["pastAC"]) / 3
    feat_table["pastGoalDiff"] = (feat_table["pastHG"] - feat_table["pastAG"]) / 3
    feat_table["pastShotsDiff"] = (feat_table["pastHS"] - feat_table["pastAG"]) / 3
    feat_table = feat_table.fillna(0)
    num_games = feat_table.shape[0] - 10
    v_split = 450
    n_games = num_games - v_split
    feat_table.drop(['pastHC', 'pastAS', 'pastAC', 'pastHG', 'pastAG'], axis=1)
    return feat_table, n_games, num_games


en_final, en_n_games, en_num_games = handle(en_final, en_table16)
es_final, es_n_games, es_num_games = handle(es_final, es_table16)
ge_final, ge_n_games, ge_num_games = handle(ge_final, ge_table16)
fr_final, fr_n_games, fr_num_games = handle(fr_final, fr_table16)
it_final, it_n_games, it_num_games = handle(it_final, it_table16)


def split(league, n_games, num_games):
    temp = league
    X = temp[['HS', 'AS', 'HST', 'AST', 'pastGoalDiff', 'HAS', 'HDS', 'AAS', 'ADS']]
    Y = temp['Result']
    # feature_weights = [0.05, 0.01, 0.1, 0.2, 0.3, 0.14, 0.2]
    min_max_scaler = MinMaxScaler()
    X[['HS', 'AS', 'HST', 'AST']] = min_max_scaler.fit_transform(X[['HS', 'AS', 'HST', 'AST']])
    # standard_scaler = StandardScaler()
    # X[['HST', 'AST']] = standard_scaler.fit_transform(X[['HST', 'AST']])
    # X = X * feature_weights
    X_train = X[0:n_games]
    y_train = Y[0:n_games]
    X_test = X[n_games:num_games - 1]
    y_test = Y[n_games:num_games - 1]
    X_predict = X[num_games:]
    return X_train, X_test, y_train, y_test, X_predict


en_X_train, en_X_test, en_y_train, en_y_test, en_X_predict = split(en_final, en_n_games, en_num_games)
es_X_train, es_X_test, es_y_train, es_y_test, es_X_predict = split(es_final, es_n_games, es_num_games)
ge_X_train, ge_X_test, ge_y_train, ge_y_test, ge_X_predict = split(ge_final, ge_n_games, ge_num_games)
fr_X_train, fr_X_test, fr_y_train, fr_y_test, fr_X_predict = split(fr_final, fr_n_games, fr_num_games)
it_X_train, it_X_test, it_y_train, it_y_test, it_X_predict = split(it_final, it_n_games, it_num_games)

rfm = RandomForestClassifier(n_estimators=150, max_depth=6)
mlp = MLPClassifier(max_iter=1000, activation='relu', hidden_layer_sizes=(100, 100))
svc = SVC()
clf_knn = KNeighborsClassifier(n_neighbors=44)

classifiers = [clf_knn, rfm, mlp, svc]


def train(X_train, X_test, y_train, y_test):
    for clf in classifiers:
        clf.fit(X_train, np.ravel(y_train))
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"{clf.__class__.__name__}: Accuracy = {accuracy:.4f}")


print("English accuracy results:")
train(en_X_train, en_X_test, en_y_train, en_y_test)
print("Spanish accuracy results:")
train(es_X_train, es_X_test, es_y_train, es_y_test)
print("German accuracy results:")
train(ge_X_train, ge_X_test, ge_y_train, ge_y_test)
print("France accuracy results:")
train(fr_X_train, fr_X_test, fr_y_train, fr_y_test)
print("Italy accuracy results:")
train(it_X_train, it_X_test, it_y_train, it_y_test)


def test_results(X_predict, league, num_games):
    y_pred_knn = clf_knn.predict(X_predict)
    y_pred_rfm = rfm.predict(X_predict)
    y_pred_mlp = mlp.predict(X_predict)
    y_pred_svc = svc.predict(X_predict)

    this_week = league[['HomeTeam', 'AwayTeam']][num_games:]
    this_week['Result_knn'] = y_pred_knn
    this_week['Result_rfm'] = y_pred_rfm
    this_week['Result_mlp'] = y_pred_mlp
    this_week['Result_svc'] = y_pred_svc
    return this_week


def transformResultBack(row, col_name):
    if row[col_name] == 1:
        return 'H'
    elif row[col_name] == 2:
        return 'A'
    else:
        return 'D'


en_res = test_results(en_X_predict, en_final, en_num_games)
es_res = test_results(es_X_predict, es_final, es_num_games)
ge_res = test_results(ge_X_predict, ge_final, ge_num_games)
fr_res = test_results(fr_X_predict, fr_final, fr_num_games)
it_res = test_results(it_X_predict, it_final, it_num_games)


def print_results(this_week):
    this_week["Res_knn"] = this_week.apply(lambda row: transformResultBack(row, "Result_knn"), axis=1)
    this_week["Res_rfm"] = this_week.apply(lambda row: transformResultBack(row, "Result_rfm"), axis=1)
    this_week["Res_mlp"] = this_week.apply(lambda row: transformResultBack(row, "Result_mlp"), axis=1)
    this_week["Res_svc"] = this_week.apply(lambda row: transformResultBack(row, "Result_svc"), axis=1)
    this_week.drop(["Result_knn", "Result_rfm", "Result_mlp", "Result_svc"], axis=1, inplace=True)
    print(this_week)


print_results(en_res)
print_results(es_res)
print_results(ge_res)
print_results(fr_res)
print_results(it_res)
