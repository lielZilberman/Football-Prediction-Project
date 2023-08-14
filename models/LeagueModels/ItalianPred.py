import pandas as pd
from models.HandleData import HandleData


it_data_folder = 'italian-serie-a_zip'

it_data = []

HandleData = HandleData()

it_merge_data = HandleData.org_data(it_data, it_data_folder)

it_res16, it_table16, it_feature_table = HandleData.compute(it_merge_data)

it_feature_table = HandleData.add_features(it_feature_table, it_table16)

it_feature_table["Result"] = it_feature_table.apply(lambda row: HandleData.transformResult(row), axis=1)

it_final = it_res16.sort_index(ascending=False)

it_final = it_final[['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']]

it_new_fixtures = pd.DataFrame([['Udinese', 'Juventus', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=it_final.columns)

it_final = HandleData.new_table(it_new_fixtures, it_final)

it_final, it_n_games, it_num_games = HandleData.handle(it_final, it_table16)

it_X_train, it_X_test, it_y_train, it_y_test, it_X_predict = HandleData.split(it_final, it_n_games, it_num_games)


print("Italy accuracy results:")
HandleData.train(it_X_train, it_X_test, it_y_train, it_y_test)


it_res = HandleData.test_results(it_X_predict, it_final, it_num_games)


HandleData.print_results(it_res)
