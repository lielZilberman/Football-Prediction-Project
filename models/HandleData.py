import os

import numpy as np
import pandas as pd
from os import path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class HandleData:

    # Initialize class attributes
    def __init__(self):
        self.season_range = (17, 22)
        self.rfm = RandomForestClassifier(n_estimators=150, max_depth=6)
        self.mlp = MLPClassifier(max_iter=1000, activation='relu', hidden_layer_sizes=(100, 100))
        self.svc = SVC()
        self.clf_knn = KNeighborsClassifier(n_neighbors=44)

    # Organize and concatenate data from the data files
    def org_data(self, data, data_folder):
        base_path = os.path.dirname(os.path.abspath(__file__))  # Get the path of the current script
        for season in range(self.season_range[0], self.season_range[1] + 1):
            data_file_path = os.path.join(base_path, '..', 'data', data_folder, 'data',
                                          'season-{:02d}{:02d}_csv.csv'.format(season, season + 1))
            data.append(data_file_path)
            # data.append(
            #     'data/{}/data/season-{:02d}{:02d}_csv.csv'.format(data_folder, season, season + 1))
        data_frame = []
        for data_file in data:
            if path.exists(data_file):
                data_frame.append(pd.read_csv(data_file))
        return pd.concat(data_frame, ignore_index=True)

    def compute(self, data):
        res_16 = data.iloc[:, :28]
        res_16 = res_16.drop(['Div', 'Date'], axis=1)
        feature_table = data.iloc[:, :28]

        table_16 = pd.DataFrame(columns=('Team', 'HGS', 'AGS', 'HAS', 'AAS', 'HGC', 'AGC', 'HDS', 'ADS'))
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
        return res_16, table_16, feature_table

    # Add additional features to the table
    def add_features(self, table, table16):
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

    # Transform result labels into numeric values
    def transformResult(self, row):
        '''Converts results (H,A or D) into numeric values'''
        if row.FTR == 'H':
            return 1
        elif row.FTR == 'A':
            return 2
        else:
            return 0

    # Create a new feature table by combining existing tables
    def new_table(self, fixtures, feat_table):
        new_feat_table = pd.concat([fixtures, feat_table], ignore_index=True)
        new_feat_table = new_feat_table.sort_index(ascending=False)
        new_feat_table = new_feat_table.reset_index().drop(['index'], axis=1)
        new_feat_table = new_feat_table.sort_index(ascending=False)
        feat_table = new_feat_table
        return feat_table

    # Handle and process feature table data
    def handle(self, feat_table, table_16):
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
            f_HAS.append(table_16[table_16['Team'] == row['HomeTeam']]['HAS'].values[0])
            f_HDS.append(table_16[table_16['Team'] == row['HomeTeam']]['HDS'].values[0])
            f_AAS.append(table_16[table_16['Team'] == row['HomeTeam']]['AAS'].values[0])
            f_ADS.append(table_16[table_16['Team'] == row['HomeTeam']]['ADS'].values[0])
        feat_table['HAS'] = f_HAS
        feat_table['HDS'] = f_HDS
        feat_table['AAS'] = f_AAS
        feat_table['ADS'] = f_ADS

        feat_table = feat_table.drop(['FTHG', 'FTAG'], axis=1)  # , 'HC', 'AC'

        feat_table["Result"] = feat_table.apply(lambda row: self.transformResult(row), axis=1)
        feat_table.sort_index(inplace=True)
        feat_table["pastCornerDiff"] = (feat_table["pastHC"] - feat_table["pastAC"]) / 3
        feat_table["pastGoalDiff"] = (feat_table["pastHG"] - feat_table["pastAG"]) / 3
        feat_table["pastShotsDiff"] = (feat_table["pastHS"] - feat_table["pastAG"]) / 3
        feat_table = feat_table.fillna(0)
        num_games = feat_table.shape[0] - 1
        v_split = 350
        n_games = num_games - v_split
        feat_table.drop(['pastHC', 'pastAS', 'pastAC', 'pastHG', 'pastAG'], axis=1)
        return feat_table, n_games, num_games

    # Split data into training and testing sets
    def split(self, league, n_games, num_games):
        temp = league
        X = temp[['HS', 'AS', 'HST', 'AST', 'pastGoalDiff', 'HAS', 'HDS', 'AAS', 'ADS', 'HC', 'AC']]
        Y = temp['Result']
        # min_max_scaler = MinMaxScaler()
        # X[['HST', 'AST']] = min_max_scaler.fit_transform(X[['HST', 'AST']])
        standard_scaler = StandardScaler()
        X.loc[:, ['HST', 'AST']] = standard_scaler.fit_transform(X[['HST', 'AST']])
        # X[['HST', 'AST']] = min_max_scaler.fit_transform(X[['HST', 'AST']])
        X_train = X[0:n_games]
        y_train = Y[0:n_games]
        X_test = X[n_games:num_games - 1]
        y_test = Y[n_games:num_games - 1]
        X_predict = X[num_games:]
        return X_train, X_test, y_train, y_test, X_predict

    # Transform numeric result values back into labels
    def transformResultBack(self, row, col_name):
        if row[col_name] == 1:
            return 'H'
        elif row[col_name] == 2:
            return 'A'
        else:
            return 'D'

    # Train different classifiers and evaluate accuracy
    def train(self, X_train, X_test, y_train, y_test):
        classifiers = [self.clf_knn, self.rfm, self.mlp, self.svc]
        for clf in classifiers:
            clf.fit(X_train, np.ravel(y_train))
            predictions = clf.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            print(f"{clf.__class__.__name__}: Accuracy = {accuracy:.4f}")

    # Make predictions using trained classifiers
    def test_results(self, X_predict, league, num_games):
        y_pred_knn = self.clf_knn.predict(X_predict)
        y_pred_rfm = self.rfm.predict(X_predict)
        y_pred_mlp = self.mlp.predict(X_predict)
        y_pred_svc = self.svc.predict(X_predict)

        this_week = league[['HomeTeam', 'AwayTeam']][num_games:]
        this_week['Result_knn'] = y_pred_knn
        this_week['Result_rfm'] = y_pred_rfm
        this_week['Result_mlp'] = y_pred_mlp
        this_week['Result_svc'] = y_pred_svc
        return this_week

    # Print the results of predictions
    def print_results(self, this_week):
        this_week["Res_knn"] = this_week.apply(lambda row: self.transformResultBack(row, "Result_knn"), axis=1)
        this_week["Res_rfm"] = this_week.apply(lambda row: self.transformResultBack(row, "Result_rfm"), axis=1)
        this_week["Res_mlp"] = this_week.apply(lambda row: self.transformResultBack(row, "Result_mlp"), axis=1)
        this_week["Res_svc"] = this_week.apply(lambda row: self.transformResultBack(row, "Result_svc"), axis=1)
        this_week.drop(["Result_knn", "Result_rfm", "Result_mlp", "Result_svc"], axis=1, inplace=True)
        print(this_week)
