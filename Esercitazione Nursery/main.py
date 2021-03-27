import pandas as pd
from SIAExam.Utils.my_utils import encode_labels, display_corr_with_col, clean_file, \
    get_train_test, classify_classifiers, display_models_score, display_corr_matrix

if __name__ == '__main__':
    # Clean file before import
    # clean_file('nursery data.csv')

    # Dataset loading
    df = pd.read_csv('nursery data.csv', sep=',')

    # Informations about data frame
    pd.set_option('max_columns', None)
    print('##################### SHAPE ###################')
    print(df.shape, '\n\n')
    print('##################### HEAD ###################')
    print(df.head(), '\n\n')
    print('################### DESCRIBE #################')
    print(df.describe(), '\n\n')

    # See what unique types we have for each column
    for col in df.columns:
        print(f'## Column {col}')
        print(df[col].unique(), '\n')

    # Preprocessing data
    # Encode all strings
    new_df = df.copy()
    encode_labels(new_df)
    print('##################### HEAD ###################')
    print(new_df.head(), '\n\n')
    print('################### DESCRIBE #################')
    print(new_df.describe(), '\n\n')

    # Data analysis
    display_corr_matrix(new_df)

    # Shows an histogram with correlation between target attribute
    # and other attributes
    display_corr_with_col(new_df, 'class')

    target = 'class'
    others = list(new_df.columns)
    others.remove(target)
    x_train, y_train, x_test, y_test = get_train_test(new_df, target, others)
    # print(x_test,'\n',x_train,'\n',y_train,'\n',y_test,'\n')
    """
    from sklearn.model_selection import train_test_split
    a, b = train_test_split(new_df, train_size=0.3)
    """
    # Train all selected classifier models with passed data
    classifiers = classify_classifiers(x_train, y_train, x_test, y_test, n_classifiers=7)

    # Display analysis of accuracy in used classifiers
    best_model_name = display_models_score(classifiers)

    from SIAExam.Utils.my_utils import validate_model
    validate_model(best_model_name, x_train, y_train, cv_score=10, cv_pred=5)

    """
    scores_df = pd.DataFrame({
        'classifier': [key for key in classifiers.keys()],
        'score': [classifiers[key]['test_score'] * 100 for key in classifiers.keys()]
    })
    print(scores_df.sort_values(by='test_score', ascending=False))
    """
    """
    # Analyze the best model we got
    gbc = GradientBoostingClassifier(n_estimators=1000)  # Same model
    scores = cross_val_score(gbc, x_train, y_train, scoring='accuracy')

    # See output
    print('Scores: ', scores, '\n')
    print('Mean: ', scores.mean(), '\n')
    print('Standard deviation: ', scores.std())

    # Check confusion matrix
    predictions = cross_val_predict(gbc, x_train, y_train)
    matrix = confusion_matrix(y_train, predictions)
    print(matrix)

    # Precision and recall
    print('Precision: ', precision_score(y_train, predictions, average='weighted'), '\n')
    print('Recall: ', recall_score(y_train, predictions, average='weighted'))
    """
