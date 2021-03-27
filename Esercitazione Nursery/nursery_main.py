'''

Un’azienda di collocamento possiede una banca dati contenente le informazioni relative a 50000 infermiere.

Si vogliono contattare tramite posta tutte le infermiere appartenenti alla classe “priority”.

Tali infermiere saranno tutte inviate ad un processo di selezione degli ospedali che hanno fatto richiesta di nuove infermiere.

L’operazione di invio delle lettere ha un costo fisso di 10,000 € e un costo individuale (per ogni infermiera contattata) pari a 5 €.

Gli ospedali prenderanno in prova le infermiere selezionate dalla ditta e alla fine del periodo di prova pagheranno alla ditta 10 € per
ogni infermiera che risulterà essere effettivamente una infermiera classificabile come appartenente alla classe “priority”.

Decidere quale modello, tra quelli studiati permette di ottenere potenzialmente il profitto maggiore.

'''
import fileinput
import time
import numpy as np
from pandas import array
from sklearn import tree
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)


def encode(data):
    for col in data.columns:
        le = LabelEncoder()
        unique_val = data[col].unique()
        le.fit(unique_val)
        all_values = data[col].values
        encoded_values = le.transform(all_values)
        data[col] = encoded_values


def clean_data(file):
    with fileinput.FileInput(file, inplace=True, backup='.bak') as f:
        for line in f:
            print(line.replace(';', ''), end='')


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

def main():
    # Pulizia dei dati
    clean_data('nursery data.csv')

    # Caricamento dataset
    data = pd.read_csv("nursery data.csv")


    print(data.head(10))
    print(data.describe())

    # Analisi degli attributi
    for col in data.columns:
        print(f'## Column {col}')
        print(data[col].unique(), '\n')

    # Pulizia degli outliers
    data['form'] = data['form'].replace('complete', 'completed')
    data['class'] = data['class'].replace('recommend','very_recom')
    print("New colimn form: \n", data['form'].unique())

    # Encoding dei valori stringa in interi

    preprocessed_data = data.copy(deep=True)
    encode(preprocessed_data)
    print(preprocessed_data.head())
    print(preprocessed_data.describe())

    # Analisi dei dati, creazione della matrice di correlazione

    corr_mat = preprocessed_data.corr()
    sn.heatmap(corr_mat, annot=True, square=True, fmt='.2f')
    plt.title('Matrice di correlazione tra gli attributi')
    plt.show()

    # Divisione in train e test set

    X=preprocessed_data
    Y=X.pop('class').values

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.3, random_state=0)


    # Allenamento e analisi dei risultati
    clss = train_classifiers(X_train, Y_train, X_test, Y_test, 5, True)

    pretty_print_classifiers_sorted(clss, "test_score")

    # K-fold validation (k=10) del classificatore migliore
    gbc = GradientBoostingClassifier(n_estimators=1000)
    print(Y_train.__class__, '\n', Y_train, '\n\n\n', Y_test.__class__)


    count = {}
    for elem in data["class"]:
        if elem in count:
            count[elem] += 1
        else:
            count[elem] = 1
    print(count)

    scores = cross_val_score(gbc, X, Y, scoring='accuracy', cv=10)

    print(scores)
    print('Media : ', scores.mean())
    print('Deviazione standard : ', scores.std())

    # Matrice di confusione
    predictions = cross_val_predict(gbc, X, Y,cv=10)
    matrix = confusion_matrix(Y, predictions)
    print(matrix)

    # Precisione e recall
    print('Precision : ', precision_score(Y, predictions, average=None), '\n')
    print('Recall : ', recall_score(Y, predictions, average=None))


if __name__ == '__main__':
    main()
