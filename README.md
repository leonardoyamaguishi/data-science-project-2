# Disaster Response Pipeline Project

## About this project:
This is the second project from Udacity's Data Science Nanodegree. The objective is to develop an Extract Transform and Load pipeline and a Machine Learning pipeline for NLP.

## Installations:
* Python 3.9.1 [1]
* sys [2]
* Pandas [3]
* NumPy [4]
* SQLite3 [5]
* Pickle [6]
* Natural Language Toolkit [7]
* Scikit-learn [8]
* Jupyter Notebook [9]
* plotly [10]
* Joblib [11]
* SQLAlchemy [12]
* Flask [13]

## Folder descriptions:
**app:** This folder contains the Flask app and the html pages.

**data:** This folder contains the CSV raw files, the SQL database file generated after processing data and the script for processing data.

**draft_notebooks:** This folder contains a Draft Notebooks in which the web app plots were based.

**models** This folder contains the script for training the classifier. **Note:** The .pkl file generated during the developmet of this project wasn't tracked on Git, as it exceeded the 100 mb limit.

## How to interact with this project:
**Data processing:** To process data, run the *process_data.py* script in the *data* folder. script requires the 3 arguments listed below in order to merge and clean the raw csv files and create a SQL database.
* filepath for disaster_messages.csv
* filepath for disaster_categories.csv
* filepath including the .db file name for the outputted SQL database

**Model training:** To train a MultiOutputClassifier with Random Forest Classifiers, run the *train_classifier.py* script in the *models*. This script requires the following arguments:
* filepath for the SQL database created with the *process_data.py* script
* filepath including the .pkl file name for the outputted trained model. In case a trained model already exists in the informed filepath, the trained model will be loaded, without training a new model.

**Web App:** To start the web app, run the *run.py* script in the *app* folder. A web app http address will be printed in the terminal. In this web app 2 visualization and a input based prediction feature were added.
* Visualization 1: the count of each category in the processed database.
* Visualization 2: the feature importance for each of the models trained with the MultiOutputClassifier.
* Input based prediction: a predicion applying the trained model will be performed based on the message inputed in the top of the page. The predicted categories will be highlighted in green.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
