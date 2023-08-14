import os
import pandas as pd
from models.HandleData import HandleData


en_data_folder = 'english-premier-league_zip'

en_data = []

HandleData = HandleData()

en_merge_data = HandleData.org_data(en_data, en_data_folder)

en_res16, en_table16, en_feature_table = HandleData.compute(en_merge_data)

en_feature_table = HandleData.add_features(en_feature_table, en_table16)

en_feature_table["Result"] = en_feature_table.apply(lambda row: HandleData.transformResult(row), axis=1)

en_final = en_res16.sort_index(ascending=False)

en_final = en_final[['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']]

en_new_fixtures = pd.DataFrame([['Southampton', 'Liverpool', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=en_final.columns)

en_final = HandleData.new_table(en_new_fixtures, en_final)

en_final, en_n_games, en_num_games = HandleData.handle(en_final, en_table16)

en_X_train, en_X_test, en_y_train, en_y_test, en_X_predict = HandleData.split(en_final, en_n_games, en_num_games)

print("English accuracy results:")
HandleData.train(en_X_train, en_X_test, en_y_train, en_y_test)

en_res = HandleData.test_results(en_X_predict, en_final, en_num_games)


HandleData.print_results(en_res)
