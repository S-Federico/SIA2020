import fileinput
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

from sklearn import tree
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Nearest Neighbors': KNeighborsClassifier(),
    'Linear SVM': SVC(),
    'Gradient Boosting Classifier': GradientBoostingClassifier(n_estimators=1000),
    'Decision Tree': tree.DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=1000),
    'Naive Bayes': GaussianNB(),
    'Ada Boost': AdaBoostClassifier(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'Gaussian Process': GaussianProcessClassifier()
}

def encode(data):
    for col in data.columns:
        le = LabelEncoder()
        unique_val = data[col].unique()
        le.fit(unique_val)
        all_values = data[col].values
        encoded_values = le.transform(all_values)
        data[col] = encoded_values

def clean_data(file): #per la nursery
    with fileinput.FileInput(file, inplace=True, backup='.bak') as f:
        for line in f:
            print(line.replace(';', ''), end='')

def train_classifiers(x_train, y_train, x_test, y_test, n_classifiers, verbose):
    models = {}
    for classifier_name, classifier in list(classifiers.items())[:n_classifiers]:
        t_start = time.process_time()
        classifier.fit(x_train, y_train)
        t_end = time.process_time()

        train_time = t_end - t_start
        train_score = classifier.score(x_train, y_train)
        test_score = classifier.score(x_test, y_test)

        models[classifier_name] = {
            'model': classifier,
            'train_score': train_score,
            'test_score': test_score,
            'train_time': train_time
        }
        if verbose:
            print(f'Trained {classifier_name} in {train_time}')
    return models

def pretty_print_classifiers_sorted(models, sortby):
    c = [key for key in models.keys()]
    test_scores = [models[key]['test_score'] for key in models.keys()]
    train_scores = [models[key]['train_score'] for key in models.keys()]
    train_times = [models[key]['train_time'] for key in models.keys()]

    models_data = {
        'classifier': c,
        'test_score': test_scores,
        'train_score': train_scores,
        'training_time': train_times
    }

    cls_df = pd.DataFrame(models_data, columns=models_data.keys())

    print(cls_df.sort_values(by=sortby, ascending=False))
    model_name = cls_df.sort_values(by=sortby, ascending=False).iloc[0]['classifier']
    return model_name


def validate_model(model_name, x_train, y_train, cv_score=10, cv_pred=5) -> None:

    # Analizziamo il miglior medello
    model = classifiers[model_name]
    scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv_score)

    # Output
    print('############ Best model:          ############', '\n', model_name, '\n')
    print('############ Model score          ############')
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Standard deviation: ', scores.std(), '\n')

    print('############ confusion matrix     ############')
    # Matrice di confusione
    predictions = cross_val_predict(model, x_train, y_train, cv=cv_pred)
    matrix = confusion_matrix(y_train, predictions)
    print(matrix, '\n')

    print('############ Precision and recall ############')
    # Precisione e recall
    print('Precision: ', precision_score(y_train, predictions, average='weighted'))
    print('Recall: ', recall_score(y_train, predictions, average='weighted'))