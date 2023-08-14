import pandas as pd
from models.HandleData import HandleData

ge_data_folder = 'german-bundesliga_zip'

ge_data = []

HandleData = HandleData()

ge_merge_data = HandleData.org_data(ge_data, ge_data_folder)

ge_res16, ge_table16, ge_feature_table = HandleData.compute(ge_merge_data)

gr_feature_table = HandleData.add_features(ge_feature_table, ge_table16)

ge_feature_table["Result"] = ge_feature_table.apply(lambda row: HandleData.transformResult(row), axis=1)

ge_final = ge_res16.sort_index(ascending=False)

ge_final = ge_final[['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']]

ge_new_fixtures = pd.DataFrame([['Wolfsburg', 'Hertha', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=ge_final.columns)

ge_final = HandleData.new_table(ge_new_fixtures, ge_final)

ge_final, ge_n_games, ge_num_games = HandleData.handle(ge_final, ge_table16)

ge_X_train, ge_X_test, ge_y_train, ge_y_test, ge_X_predict = HandleData.split(ge_final, ge_n_games, ge_num_games)


print("German accuracy results:")
HandleData.train(ge_X_train, ge_X_test, ge_y_train, ge_y_test)


ge_res = HandleData.test_results(ge_X_predict, ge_final, ge_num_games)


HandleData.print_results(ge_res)
