import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from IPython.display import display
from os import path, makedirs, walk
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

en_data_folder = 'english-premier-league_zip'
es_data_folder = 'spanish-la-liga_zip'
fr_data_folder = 'french-ligue-1_zip'
ge_data_folder = 'german-bundesliga_zip'
it_data_folder = 'italian-serie-a_zip'

data_folders = [en_data_folder, es_data_folder,
                fr_data_folder, ge_data_folder, it_data_folder]
season_range = (9, 22)
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

input_cols = ['home_encoded', 'away_encoded', 'HS', 'AS', 'HST', 'AST']
# input_cols = ['home_encoded', 'away_encoded', 'HTHG', 'HTAG', 'HS',
#                 'AS', 'HST', 'AST', 'HR', 'AR']
output_cols = ['FTR']
all_cols = input_cols + output_cols


def process_data(data):
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
    data = data[all_cols]

    data = data.dropna(axis=0)
    return data


en_merge_data = process_data(en_merge_data)
es_merge_data = process_data(es_merge_data)
ge_merge_data = process_data(ge_merge_data)
fr_merge_data = process_data(fr_merge_data)
it_merge_data = process_data(it_merge_data)


# print(it_merge_data)


def train_classifier(clf, X_train, y_train):
    clf.fit(X_train, y_train)


def predict_labels(clf, features, target):
    y_pred = clf.predict(features)
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
    f1, acc = predict_labels(clf, X_test, y_test)
    print("Test Metrics:")
    print("-" * 20)
    print("F1 Score:{}".format(f1))
    print("Accuracy:{}".format(acc))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, clf.predict(X_test, )))


Xen = en_merge_data[input_cols]
print(Xen)
Yen = en_merge_data['FTR']

Xen_train, Xen_test, Yen_train, Yen_test = train_test_split(Xen, Yen, test_size=0.2, random_state=2, stratify=Yen)

svc_classifier = SVC(random_state=100, kernel='rbf')
lr_classifier = LogisticRegression(multi_class='ovr', max_iter=500)
nbClassifier = GaussianNB()
rfClassifier = RandomForestClassifier()

##
print()
print("Logistic Regression one vs All Classifier")
print("-" * 20)
model(lr_classifier, Xen_train, Yen_train, Xen_test, Yen_test)

print()
print("Gaussain Naive Bayes Classifier")
print("-" * 20)
model(nbClassifier, Xen_train, Yen_train, Xen_test, Yen_test)

print()
print("Random Forest Classifier")
print("-" * 20)
model(rfClassifier, Xen_train, Yen_train, Xen_test, Yen_test)
