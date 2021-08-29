import sys

# Libraries for data handling
import pandas as pd
import numpy as np
import sqlite3
import pickle

# Modules for data processing
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

# Modules for feature extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Modules for modelselection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Modules for model fitting
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Modules for metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Modules for data processing
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

# Modules for feature extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Modules for modelling
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Modules for metrics
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - filepath to the db file created on process_data.py on the save_data function

    OUTPUTS: 
    X - Model input ndarray
    Y - Model outputs ndarray
    category_names - list
    '''
    
    # Connect to the given database and run a SQLite command to retrieve data from the database
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse', conn)
    
    # Split the database in X (model inputs) and Y (model outputs), both as ndarray outputs
    columns = list(df.columns)
    X = df['message'].values
    Y = df[columns[4::]].values
    
    # List the category names from Y
    category_names = np.array(columns[4::])
    
    return X, Y, category_names

def tokenize(text):
    '''
    INPUTS:
    text - a string to be tokenized
    OUTPUTS:
    clean_tokens - list of clean tokens
    '''

    # Tokenizing the text
    tokenized_text = word_tokenize(text)
    
    # Starts the lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Creates a list of lemmatized tokens
    clean_tokens = []
    for tok in tokenized_text:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
    '''
    INPUTS:
    None
    
    OUTPUTS:
    model - This function outputs a MultiOutputClassfier to be trained using the fit method.
    The MultiOutputClassifier is based on a RandomForestClassifier estimator.
    The parameter selection is otbained through GridSearchCV based on parameters from pipeline_tfidf_param.
    The parameter selection was limited for demostration purposes, avoiding timeouts.
    '''
    
    pipeline_tfidf_rfc = Pipeline([
        ('transformer', TfidfVectorizer(tokenizer = tokenize, max_features = 10000)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    pass

    pipeline_tfidf_param = {
        'transformer__stop_words': ['english', None]
    }
    
    return GridSearchCV(pipeline_tfidf_rfc, param_grid = pipeline_tfidf_param, n_jobs = -1)

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUTS:
    model - Trained model for prediction
    X_test - Model inputs as ndarray
    Y_test - True results for the prediction of X_test
    category_names - Names for each category predicted by the Multi Output Classifier
    
    OUTPUTS:
    none - a classification report will be printed.
    '''
    prediction = model.predict(X_test)
    
    # Transposing the outputs to loop through the predicted classes
    y_test_t = Y_test.T.astype(int)
    y_pred_t = prediction.T.astype(int)
    
    # Looping through the predicted classes, incrementing the index in order to display the class name in the report
    index = 0
    
    for col_true, col_pred in zip(y_test_t, y_pred_t):
        
        # As it is possible to have non binary outputs, the correction below is needed in order to print the reports correctly
        label_correction = np.vectorize(lambda x: str(category_names[index]) if x > 0 else 'None')
        col_true = label_correction(col_true)
        col_pred = label_correction(col_pred)
        
        print(classification_report(col_true, col_pred))
        
        # Category name index incrementation
        index = index + 1
    
    pass
        
def save_model(model, model_filepath):
    '''
    INPUTS:
    model - Model to be saved
    model_filepath - Filepath including model name and format
    
    OUTPUTS:
    None - saves the model in the informed filepath 
    '''

    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        try:
            print('Loading model...')
            print(model_filepath)
            model = pickle.load(open(model_filepath, 'rb'))
            print('Model found...')
            
        except:
            print('Model not found...')
            print('Building model...')
            model = build_model()
            
            print('Training model...')
            model.fit(X_train, Y_train)

            print('Saving model...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)
            print('Trained model saved!')
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()