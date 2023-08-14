import pandas as pd
from models.HandleData import HandleData

es_data_folder = 'spanish-la-liga_zip'

es_data = []

HandleData = HandleData()

es_merge_data = HandleData.org_data(es_data, es_data_folder)

es_res16, es_table16, es_feature_table = HandleData.compute(es_merge_data)

es_feature_table = HandleData.add_features(es_feature_table, es_table16)

es_feature_table["Result"] = es_feature_table.apply(lambda row: HandleData.transformResult(row), axis=1)

es_final = es_res16.sort_index(ascending=False)

es_final = es_final[['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC']]

es_new_fixtures = pd.DataFrame([['Valladolid', 'Getafe', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=es_final.columns)

es_final = HandleData.new_table(es_new_fixtures, es_final)

es_final, es_n_games, es_num_games = HandleData.handle(es_final, es_table16)

es_X_train, es_X_test, es_y_train, es_y_test, es_X_predict = HandleData.split(es_final, es_n_games, es_num_games)


print("Spanish accuracy results:")
HandleData.train(es_X_train, es_X_test, es_y_train, es_y_test)


es_res = HandleData.test_results(es_X_predict, es_final, es_num_games)


HandleData.print_results(es_res)
