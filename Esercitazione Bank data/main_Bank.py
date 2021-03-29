import pandas as pd
import seaborn as sn
import arff
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Utils_esercitazioni.Classification_utils import train_classifiers, pretty_print_classifiers_sorted, validate_model, \
    encode

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', None)


def main():
    # Lettura del file
    bank = pd.read_csv('bank-data.csv', sep=',')
    print(bank.head())
    print(bank.describe())

    # Drop dei duplicati
    bank.drop_duplicates(subset=['id'], inplace=True, keep='first')

    # Rimuoviamo il campo id e salviamo in bakn-data.arff
    del bank['id']
    arff.dump('bank-data.arff', bank.values, relation='bank_data', names=bank.columns)

    # Discretizziamo 'age' in 10 bins e 'children' e salviamo in bank-data-1.arff
    pd.cut(bank['age'], 10)
    pd.cut(bank['children'], [0, 1, 2, 3])
    print('##################### HEAD ###################')
    print(bank.head())
    arff.dump('bank-data-1.arff', bank.values, relation='bank_data', names=bank.columns)

    # Droppiamo le colonne già analizzate da bank e facciamo l'encoding degli attributi categoriali
    encode(bank)

    # Mostriamo la matrice di correlazione e la correlazione come istogrammma
    corr_matrix = bank.corr()
    sn.heatmap(corr_matrix, annot=True, square=True, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
    sn.jointplot(x='income', y='children', data=bank)
    plt.show()

    # Preparazione dei set per il training

    X = bank
    Y = X.pop('pep')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    # Training dei modelli
    classifiers = train_classifiers(x_train, y_train, x_test, y_test,10,True)

    # Print delle performance
    best_model_name = pretty_print_classifiers_sorted(classifiers,'test_score')
    print(best_model_name)
    # Analisi modello migliore
    validate_model(best_model_name, x_train, y_train)


if __name__ == '__main__':
    main()
