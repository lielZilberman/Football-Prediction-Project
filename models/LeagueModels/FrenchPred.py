import pandas as pd
from models.HandleData import HandleData

fr_data_folder = 'french-ligue-1_zip'

fr_data = []

HandleData = HandleData()

fr_merge_data = HandleData.org_data(fr_data, fr_data_folder)

fr_res16, fr_table16, fr_feature_table = HandleData.compute(fr_merge_data)

fr_feature_table = HandleData.add_features(fr_feature_table, fr_table16)

fr_feature_table["Result"] = fr_feature_table.apply(lambda row: HandleData.transformResult(row), axis=1)

fr_final = fr_res16.sort_index(ascending=False)

fr_final = fr_final[['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']]

fr_new_fixtures = pd.DataFrame([['Troyes', 'Lille', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=fr_final.columns)

fr_final = HandleData.new_table(fr_new_fixtures, fr_final)

fr_final, fr_n_games, fr_num_games = HandleData.handle(fr_final, fr_table16)

fr_X_train, fr_X_test, fr_y_train, fr_y_test, fr_X_predict = HandleData.split(fr_final, fr_n_games, fr_num_games)


print("France accuracy results:")
HandleData.train(fr_X_train, fr_X_test, fr_y_train, fr_y_test)


fr_res = HandleData.test_results(fr_X_predict, fr_final, fr_num_games)


HandleData.print_results(fr_res)
