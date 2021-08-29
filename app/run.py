import json
import plotly
import pandas as pd
import joblib
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

from sqlalchemy import create_engine
import sqlite3

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
#OLD
# engine = create_engine('sqlite:///../data/DisasterResponse.db')
# df = pd.read_sql_table('DisasterResponse', engine)

conn = sqlite3.connect('data/DisasterResponse.db')
df = pd.read_sql('SELECT * FROM DisasterResponse', conn)

# load model
# OLD
# model = joblib.load("../models/classifier.pkl")

model = joblib.load(open('models/classifier.pkl', 'rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    '''
    Plot 1 - Bar Chart with the count per category in the model dataset
    '''

    # Listing category names
    category_names = df.columns[4:]

    # Counting the occurence of each category
    # Filtering the Category Names from the DataFrame, converting to np array and summing
    category_count = df[category_names].to_numpy().sum(axis = 0)
    
    '''
    Plot 2 - Horizontal Bar Chart with the Feature Importance of words
    '''
    # Retrieving the Random Forest Classifier
    pipeline_estimators = model.best_estimator_['clf']
    # Retrieving a list of the estimators that compose the Random Forest Classifier
    rfc_estimators = pipeline_estimators.estimators_

    # Retrieving the Model Vocabulary to label the features
    model_vocabulary = model.best_estimator_['transformer'].vocabulary_
    model_vocabulary_keys = model_vocabulary.keys()

    # Creating a DataFrame with the Feature Importance for each classifier
    feature_importance_dict = {}

    for i in range(0,len(category_names)):
        feature_importance_dict[i] = rfc_estimators[i].feature_importances_

    feature_importance_df = pd.DataFrame(feature_importance_dict, index = model_vocabulary_keys)

    # Inserting buttons for each trained classifier
    # The buttons must refresh the visualization inserting the top 40 most important features for the classification

    buttons = []

    for i in range(0,len(category_names)):
        buttons.append(dict(method='update',
                            label=category_names[i],
                            visible=True,
                            args=[{'y':[feature_importance_df[feature_importance_df.columns[i]].sort_values(ascending = False)[0:40].index],
                                'x':[feature_importance_df[feature_importance_df.columns[i]].sort_values(ascending = False)[0:40]],
                                'type':'bar'}, [0]],
                            )
                    )

    # Setting the update menu
    updatemenu = [{'buttons': buttons,
                'direction' : 'down',
                'showactive' : True
                }]

    '''
    Create visuals
    '''

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_count
                )
            ],

            'layout': {
                'title': 'Distribution of Category Count',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=feature_importance_df[feature_importance_df.columns[0]].sort_values(ascending = False)[0:40], 
                    y=feature_importance_df[feature_importance_df.columns[0]].sort_values(ascending = False)[0:40].index, 
                    orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Trained model word feature importance',
                'yaxis': {
                    'title': "Word"
                },
                'xaxis': {
                    'title': "Feature Importance"
                },
                'showlegends' : False,
                'updatemenus' : updatemenu
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()