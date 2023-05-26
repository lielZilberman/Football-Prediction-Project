import json

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, BaggingClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from time import time
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from os import path, makedirs, walk
from joblib import dump, load
from xgboost import XGBClassifier


# Utility Functions


def train_classifier(clf, X_train, y_train):
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    # print("Model trained in {:2f} seconds".format(end - start))


def predict_labels(clf, features, target):
    start = time()
    y_pred = clf.predict(features)
    end = time()
    # print("Made Predictions in {:2f} seconds".format(end - start))

    acc = sum(target == y_pred) / float(len(y_pred))

    return f1_score(target, y_pred, average='micro'), acc


def model(clf, X_train, y_train, X_test, y_test):
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    train_classifier(clf, X_train, y_train)

    f1, acc = predict_labels(clf, X_train, y_train)
    print("Training Info:")
    print("-" * 20)
    print("F1 Score:{}".format(f1))
    print("Accuracy:{}".format(acc))
    # print("Confusion Matrix:\n")
    # print(confusion_matrix(y_train, clf.predict(X_train)))

    f1, acc = predict_labels(clf, X_test, y_test)
    print("Test Metrics:")
    print("-" * 20)
    print("F1 Score:{}".format(f1))
    print("Accuracy:{}".format(acc))
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test, clf.predict(X_test,)))


def derive_clean_sheet(src):
    arr = []
    n_rows = src.shape[0]

    for data in range(n_rows):

        # [HTHG, HTAG]
        values = src.iloc[data][['HTHG', 'HTAG']].values
        if pd.isna(values).any():
            continue

        cs = [0, 0]

        if values[0] == 0:
            cs[1] = 1

        if values[1] == 0:
            cs[0] = 1

        arr.append(cs)

    return arr


# Data gathering

en_data_folder = 'english-premier-league_zip'
es_data_folder = 'spanish-la-liga_zip'
fr_data_folder = 'french-ligue-1_zip'
ge_data_folder = 'german-bundesliga_zip'
it_data_folder = 'italian-serie-a_zip'

# data_folders = [es_data_folder]
data_folders = [en_data_folder, es_data_folder,
                fr_data_folder, ge_data_folder, it_data_folder]

season_range = (9, 22)

data_files = []
for data_folder in data_folders:
    for season in range(season_range[0], season_range[1] + 1):
        data_files.append(
            'data/{}/data/season-{:02d}{:02d}_csv.csv'.format(data_folder, season, season + 1))

data_frames = []

for data_file in data_files:
    if path.exists(data_file):
        data_frames.append(pd.read_csv(data_file))

data = pd.concat(data_frames).reset_index()
print(data)


# Pre processing
input_filter = ['home_encoded', 'away_encoded', 'HTHG', 'HTAG', 'HS',
                'AS', 'HST', 'AST', 'HR', 'AR']
output_filter = ['FTR']

cols_to_consider = input_filter + output_filter

encoder = LabelEncoder()
home_encoded = encoder.fit_transform(data['HomeTeam'])
home_encoded_mapping = dict(
    zip(encoder.classes_, encoder.transform(encoder.classes_).tolist()))
data['home_encoded'] = home_encoded

encoder = LabelEncoder()
away_encoded = encoder.fit_transform(data['AwayTeam'])
away_encoded_mapping = dict(
    zip(encoder.classes_, encoder.transform(encoder.classes_).tolist()))
data['away_encoded'] = away_encoded

# Deriving Clean Sheet
# htg_df = data[['HTHG', 'HTAG']]
# cs_data = derive_clean_sheet(htg_df)
# cs_df = pd.DataFrame(cs_data, columns=['HTCS', 'ATCS'])

# data = pd.concat([data, cs_df], axis=1)

data = data[cols_to_consider]

print(data[data.isna().any(axis=1)])
data = data.dropna(axis=0)

# Training & Testing

X = data[input_filter]
Y = data['FTR']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
from sklearn.ensemble import BaggingClassifier

svc_classifier = SVC(random_state=100, kernel='rbf')
lr_classifier = LogisticRegression(multi_class='ovr', max_iter=500)
dtClassifier = DecisionTreeClassifier()
rfClassifier = RandomForestClassifier()
xgbClassifier = XGBClassifier()
adaClassifier = AdaBoostClassifier()

print("Support Vector Machine")
print("-" * 20)
model(svc_classifier, X_train, Y_train, X_test, Y_test)

print()
print("Logistic Regression one vs All Classifier")
print("-" * 20)
model(lr_classifier, X_train, Y_train, X_test, Y_test)


print()
print("Bagged Decision Tree Classifier")
print("-" * 20)
bagged_dtClassifier = BaggingClassifier(DecisionTreeClassifier(), n_estimators=10)
model(bagged_dtClassifier, X_train, Y_train, X_test, Y_test)

print()
print("Bagged Random Forest Classifier")
print("-" * 20)
bagged_rfClassifier = BaggingClassifier(RandomForestClassifier(), n_estimators=10)
model(bagged_rfClassifier, X_train, Y_train, X_test, Y_test)

print()
print("Adaboost Classifier")
print("-" * 20)
model(adaClassifier, X_train, Y_train, X_test, Y_test)

print()
print("XGBoost Classifier")
print("-" * 20)
model(xgbClassifier, X_train, Y_train, X_test, Y_test)


# Exporting the Model
print()
print()
shouldExport = input('Do you want to export the model(s) (y / n) ? ')
if shouldExport.strip().lower() == 'y':
    exportedModelsPath = 'exportedModels'

    makedirs(exportedModelsPath, exist_ok=True)

    dump(lr_classifier, f'{exportedModelsPath}/lr_classifier.model')
    dump(dtClassifier, f'{exportedModelsPath}/dt_classifier.model')
    dump(rfClassifier, f'{exportedModelsPath}/rf_classifier.model')
    dump(xgbClassifier, f'{exportedModelsPath}/xgb_classifier.model')
    dump(adaClassifier, f'{exportedModelsPath}/ada_classifier.model')

    exportMetaData = dict()
    exportMetaData['home_teams'] = home_encoded_mapping
    exportMetaData['away_teams'] = away_encoded_mapping

    exportMetaDataFile = open(f'{exportedModelsPath}/metaData.json', 'w')
    json.dump(exportMetaData, exportMetaDataFile)

    print(f'Model(s) exported successfully to {exportedModelsPath}/')










