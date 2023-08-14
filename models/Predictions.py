import pandas as pd
from models.HandleData import HandleData

en_data_folder = 'english-premier-league_zip'
es_data_folder = 'spanish-la-liga_zip'
fr_data_folder = 'french-ligue-1_zip'
ge_data_folder = 'german-bundesliga_zip'
it_data_folder = 'italian-serie-a_zip'

data_folders = [en_data_folder, es_data_folder,
                fr_data_folder, ge_data_folder, it_data_folder]
en_data = []
es_data = []
fr_data = []
ge_data = []
it_data = []

HandleData = HandleData()

en_merge_data = HandleData.org_data(en_data, en_data_folder)
es_merge_data = HandleData.org_data(es_data, es_data_folder)
ge_merge_data = HandleData.org_data(ge_data, ge_data_folder)
fr_merge_data = HandleData.org_data(fr_data, fr_data_folder)
it_merge_data = HandleData.org_data(it_data, it_data_folder)

en_res16, en_table16, en_feature_table = HandleData.compute(en_merge_data)
es_res16, es_table16, es_feature_table = HandleData.compute(es_merge_data)
ge_res16, ge_table16, ge_feature_table = HandleData.compute(ge_merge_data)
fr_res16, fr_table16, fr_feature_table = HandleData.compute(fr_merge_data)
it_res16, it_table16, it_feature_table = HandleData.compute(it_merge_data)

en_feature_table = HandleData.add_features(en_feature_table, en_table16)
es_feature_table = HandleData.add_features(es_feature_table, es_table16)
gr_feature_table = HandleData.add_features(ge_feature_table, ge_table16)
fr_feature_table = HandleData.add_features(fr_feature_table, fr_table16)
it_feature_table = HandleData.add_features(it_feature_table, it_table16)

en_feature_table["Result"] = en_feature_table.apply(lambda row: HandleData.transformResult(row), axis=1)
es_feature_table["Result"] = es_feature_table.apply(lambda row: HandleData.transformResult(row), axis=1)
ge_feature_table["Result"] = ge_feature_table.apply(lambda row: HandleData.transformResult(row), axis=1)
fr_feature_table["Result"] = fr_feature_table.apply(lambda row: HandleData.transformResult(row), axis=1)
it_feature_table["Result"] = it_feature_table.apply(lambda row: HandleData.transformResult(row), axis=1)

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
# Adding next week fixtures
en_new_fixtures = pd.DataFrame([['Southampton', 'Liverpool', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=en_final.columns)
es_new_fixtures = pd.DataFrame([['Valladolid', 'Getafe', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=es_final.columns)
ge_new_fixtures = pd.DataFrame([['Wolfsburg', 'Hertha', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=ge_final.columns)
fr_new_fixtures = pd.DataFrame([['Troyes', 'Lille', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=fr_final.columns)
it_new_fixtures = pd.DataFrame([['Udinese', 'Juventus', 'D', 0, 0, 0, 0, 0, 0, 0, 0]], columns=it_final.columns)

en_final = HandleData.new_table(en_new_fixtures, en_final)
es_final = HandleData.new_table(es_new_fixtures, es_final)
ge_final = HandleData.new_table(ge_new_fixtures, ge_final)
fr_final = HandleData.new_table(fr_new_fixtures, fr_final)
it_final = HandleData.new_table(it_new_fixtures, it_final)

en_final, en_n_games, en_num_games = HandleData.handle(en_final, en_table16)
es_final, es_n_games, es_num_games = HandleData.handle(es_final, es_table16)
ge_final, ge_n_games, ge_num_games = HandleData.handle(ge_final, ge_table16)
fr_final, fr_n_games, fr_num_games = HandleData.handle(fr_final, fr_table16)
it_final, it_n_games, it_num_games = HandleData.handle(it_final, it_table16)

en_X_train, en_X_test, en_y_train, en_y_test, en_X_predict = HandleData.split(en_final, en_n_games, en_num_games)
es_X_train, es_X_test, es_y_train, es_y_test, es_X_predict = HandleData.split(es_final, es_n_games, es_num_games)
ge_X_train, ge_X_test, ge_y_train, ge_y_test, ge_X_predict = HandleData.split(ge_final, ge_n_games, ge_num_games)
fr_X_train, fr_X_test, fr_y_train, fr_y_test, fr_X_predict = HandleData.split(fr_final, fr_n_games, fr_num_games)
it_X_train, it_X_test, it_y_train, it_y_test, it_X_predict = HandleData.split(it_final, it_n_games, it_num_games)


print("English accuracy results:")
HandleData.train(en_X_train, en_X_test, en_y_train, en_y_test)
print("Spanish accuracy results:")
HandleData.train(es_X_train, es_X_test, es_y_train, es_y_test)
print("German accuracy results:")
HandleData.train(ge_X_train, ge_X_test, ge_y_train, ge_y_test)
print("France accuracy results:")
HandleData.train(fr_X_train, fr_X_test, fr_y_train, fr_y_test)
print("Italy accuracy results:")
HandleData.train(it_X_train, it_X_test, it_y_train, it_y_test)


en_res = HandleData.test_results(en_X_predict, en_final, en_num_games)
es_res = HandleData.test_results(es_X_predict, es_final, es_num_games)
ge_res = HandleData.test_results(ge_X_predict, ge_final, ge_num_games)
fr_res = HandleData.test_results(fr_X_predict, fr_final, fr_num_games)
it_res = HandleData.test_results(it_X_predict, it_final, it_num_games)


HandleData.print_results(en_res)
HandleData.print_results(es_res)
HandleData.print_results(ge_res)
HandleData.print_results(fr_res)
HandleData.print_results(it_res)
