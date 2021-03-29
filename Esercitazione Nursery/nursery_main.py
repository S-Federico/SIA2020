"""

Un’azienda di collocamento possiede una banca dati contenente le informazioni relative a 50000 infermiere.

Si vogliono contattare tramite posta tutte le infermiere appartenenti alla classe “priority”.

Tali infermiere saranno tutte inviate ad un processo di selezione degli ospedali che hanno fatto richiesta di nuove infermiere.

L’operazione di invio delle lettere ha un costo fisso di 10,000 € e un costo individuale (per ogni infermiera contattata) pari a 5 €.

Gli ospedali prenderanno in prova le infermiere selezionate dalla ditta e alla fine del periodo di prova pagheranno alla ditta 10 € per
ogni infermiera che risulterà essere effettivamente una infermiera classificabile come appartenente alla classe “priority”.

Decidere quale modello, tra quelli studiati permette di ottenere potenzialmente il profitto maggiore.

"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from Utils_esercitazioni.Classification_utils import clean_data, encode, train_classifiers, \
    pretty_print_classifiers_sorted

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

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
    data['class'] = data['class'].replace('recommend', 'very_recom')
    print("New column form: \n", data['form'].unique())

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
    X = preprocessed_data
    Y = X.pop('class').values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    # Allenamento e analisi dei risultati
    clss = train_classifiers(X_train, Y_train, X_test, Y_test, 5, True)
    pretty_print_classifiers_sorted(clss, "test_score")

    # K-fold validation (k=10) del classificatore migliore
    gbc = GradientBoostingClassifier(n_estimators=1000)

    scores = cross_val_score(gbc, X, Y, scoring='accuracy', cv=10)

    print(scores)
    print('Media : ', scores.mean())
    print('Deviazione standard : ', scores.std())

    # Matrice di confusione
    predictions = cross_val_predict(gbc, X, Y, cv=10)
    matrix = confusion_matrix(Y, predictions)
    print(matrix)

    # Precisione e recall
    print('Precision : ', precision_score(Y, predictions, average=None), '\n')
    print('Recall : ', recall_score(Y, predictions, average=None))


if __name__ == '__main__':
    main()
