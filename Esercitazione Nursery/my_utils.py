"""
This module contains all functions that could be used in more applications.
"""
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import fileinput
import seaborn as sn
from sklearn import tree
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.svm import SVC

classifiers_dict = {
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


def clean_file(filename: str) -> None:
    """
    This function is used to remove extra ';' from csv file to make it readable

    :param filename: The name of the csv file to fix
    :return:
    """
    with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
        for line in file:
            print(line.replace(';', ''), end='')


def encode_labels(df: pd.DataFrame, columns: [str] = None) -> None:
    """
    This function helps to encode string values.
    You can specify if you want to encode all columns (not passing anything to 'columns'
    parameter) or only some columns

    :param df: Used dataframe
    :param columns: column names to encode. If you want to encode all leave it blank
    :return: None
    """
    if not columns:
        columns = df.columns
    for column in columns:
        le = LabelEncoder()
        unique_values = df[column].unique()
        le.fit(unique_values)
        all_values = df[column].values
        encoded_values = le.transform(all_values)
        df[column] = encoded_values


def display_corr_with_col(df: pd.DataFrame, col: str) -> None:
    """
    Display an histogram showing how is correlated a certain column compared to other values

    :param df: The dataframe to work on
    :param col: The name of the target column
    :return: None
    """
    corr_mat = df.corr()
    corr_type = corr_mat[col].copy()
    abs_corr_type = corr_type.apply(lambda x: abs(x))
    # We sort in descending way so that the first element will surely be 'col', that
    # will have correlation=1 (maximum correlation)
    desc_corr_values = abs_corr_type.sort_values(ascending=False)
    y_values = desc_corr_values.values[1:]
    x_values = range(0, len(y_values))
    x_labels = desc_corr_values.keys()[1:]
    plot, axis = plt.subplots()
    axis.bar(x_values, y_values)
    axis.set_title(f'Correlation between {col} and other features')
    plt.xticks(x_values, x_labels, rotation='vertical')
    plt.show()


def display_corr_matrix(df: pd.DataFrame) -> None:
    """
    This function displays the correlation matrix of a given dataframe

    :param df: Input dataframe
    :return: None
    """
    corr_matrix = df.corr()
    print(corr_matrix)
    sn.heatmap(corr_matrix, annot=True, square=True, fmt='.2f')
    plt.title('Correlation matrix between features')
    plt.show()


def get_train_test(df: pd.DataFrame, y_col: str, x_cols: [str], ratio: float = 0.7) -> tuple:
    """
    This function is used to divide data from a dataframe in train set and
    test set by passing a speific ratio to divide the data in these sets.

    :param df: The dataframe to work on
    :param y_col: The name of the target column
    :param x_cols: The name of other columns except target
    :param ratio: The ratio train (Test is usually 0.7, so 70%)
    :return: Tuple with: x_train, y_train, x_test, y_test
    """
    # Calculate a random ratio

    ratio = np.random.rand(len(df)) < ratio

    df_train = df[ratio]
    df_test = df[~ratio]
    # df_train, df_test = train_test_split(df, ratio)

    y_train = df_train[y_col].values
    y_test = df_test[y_col].values
    x_train = df_train[x_cols].values
    x_test = df_test[x_cols].values

    return x_train, y_train, x_test, y_test


def classify_classifiers(x_train, y_train, x_test, y_test, n_classifiers: int = None, verbose: bool = True) -> {}:
    """
    Train n_classifiers models to see wich is better

    :param x_train: non-target training set
    :param y_train: target training set
    :param x_test: non-target test set
    :param y_test: target test set
    :param n_classifiers: How many classifiers to test
    :param verbose: If true, more info in terminal about execution. True by defeault
    :return: Dict of models used with information
    """
    models = {}
    if not n_classifiers:
        print('No number of classifiers specified, so all classifiers will be used')
        n_classifiers = len(classifiers_dict)
    elif n_classifiers < 1 or n_classifiers > len(classifiers_dict):
        print('Invalid number of classifiers used, so all classifiers will be used')
        n_classifiers = len(classifiers_dict)
    for classifier_name, classifier in list(classifiers_dict.items())[:n_classifiers]:
        t_start = time.process_time()
        classifier.fit(x_train, y_train)
        t_end = time.process_time()

        elapsed_time = t_end - t_start
        train_score = classifier.score(x_train, y_train)
        test_score = classifier.score(x_test, y_test)

        models[classifier_name] = {
            'model': classifier,
            'train_score': train_score,
            'test_score': test_score,
            'training_time': elapsed_time
        }
        if verbose:
            print(f'Trained {classifier_name} in {elapsed_time}')
    return models


def display_models_score(models: dict, sort_by='test_score') -> str:
    """
    Display classification models results

    :param models: classifiers dict created in 'classify_classifiers'
    :param sort_by: Parameter used to sort. Default is 'test_score'
    :return: Name of the best model that can be used to retrieve the model from 'classifiers_dict'
    """
    classifiers = [key for key in models.keys()]
    test_scores = [models[key]['test_score'] for key in models.keys()]
    train_scores = [models[key]['train_score'] for key in models.keys()]
    training_times = [models[key]['training_time'] for key in models.keys()]

    m_data = {
        'classifier': classifiers,
        'test_score': test_scores,
        'train_score': train_scores,
        'training_time': training_times
    }

    """    
    cls_df = pd.DataFrame(data=np.zeros(shape=len(classifiers)), columns=[
        'classifier',
        'test_score',
        'train_score',
        'training_time'
    ])
    """

    cls_df = pd.DataFrame(m_data, columns=m_data.keys())
    """
    for i in range(0, len(classifiers)):
        cls_df.loc[i, 'classifier'] = classifiers[i]
        cls_df.loc[i, 'train_score'] = train_scores[i]
        cls_df.loc[i, 'test_score'] = test_scores[i]
        cls_df.loc[i, 'training_time'] = training_times[i]
    """
    print(cls_df.sort_values(by=sort_by, ascending=False))
    model_name = cls_df.sort_values(by=sort_by, ascending=False).iloc[0]['classifier']
    return model_name


def validate_model(model_name, x_train, y_train, cv_score=10, cv_pred=5) -> None:
    """
    This function use validation on model and prints validation results

    :param cv_pred:
    :param cv_score:
    :param model_name: The model to be validated
    :param x_train: Data used to train the model (not containing target)
    :param y_train: Target data used to train the model
    :return: None
    """
    # Analyze the best model we got
    model = classifiers_dict[model_name]
    scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv_score)

    # See output
    print('############ Best model:          ############', '\n', model_name, '\n')
    print('############ Model score          ############')
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Standard deviation: ', scores.std(), '\n')

    print('############ confusion matrix     ############')
    # Check confusion matrix
    predictions = cross_val_predict(model, x_train, y_train, cv=cv_pred)
    matrix = confusion_matrix(y_train, predictions)
    print(matrix, '\n')

    print('############ Precision and recall ############')
    # Precision and recall
    print('Precision: ', precision_score(y_train, predictions, average='weighted'))
    print('Recall: ', recall_score(y_train, predictions, average='weighted'))
