# Disaster Response Pipeline Project

## About this project:
This is the second project from Udacity's Data Science Nanodegree. The objective of this project is to develop an application to assist in the mobilization of government and humanitarian organizations in order to assess recent or occurring disasters and determine the immediate course of action by categorizing tweets and text messages compiled and provided by Figure Eight [14].

The categorization is an output from a deployed Machine Learning model obtained by following an Extract, Transform and Load pipeline and a Machine Learning pipeline with the application of sklearn's GridSearchCV for model selection. The trained model is deployed in a Flask Web App.

## Installations:
* [Python 3.9.1](https://www.python.org) [1]
* [sys](https://docs.python.org/3/library/sys.html) [2]
* [Pandas](https://pandas.pydata.org) [3]
* [NumPy](https://numpy.org) [4]
* [SQLite3](https://docs.python.org/3/library/sqlite3.html) [5]
* [Pickle](https://docs.python.org/3/library/pickle.html) [6]
* [Natural Language Toolkit](https://www.nltk.org) [7]
* [Scikit-learn](https://scikit-learn.org/stable/) [8]
* [Jupyter Notebook](https://jupyter.org) [9]
* [plotly](https://plotly.com/python/) [10]
* [Joblib](https://joblib.readthedocs.io/en/latest/) [11]
* [SQLAlchemy](https://www.sqlalchemy.org) [12]
* [Flask](https://flask.palletsprojects.com/en/2.0.x/) [13]

## Folder descriptions:
**app:** This folder contains the Flask app and the html templates.

**data:** This folder contains the CSV raw files, the SQL database file obtained after processing the data and the script for processing data.

**draft_notebooks:** This folder contains a Jupyter Notebook in which the web app plots were based.

**models** This folder contains the script for training the classifier. **Note:** The .pkl file generated during the developmet of this project wasn't tracked on Git, as it exceeded Git's file size limit. In this project, the GridSearchCV tested a limited set of parameters as the model selection could be significantly more demanding as the number of possible parameters increased, moreover, the GridSearchCV models trained with more parameters in the local machine presented a file size that exceeded Git's file size limit by a significant amount.

## How to interact with this project:
**Data processing:** To process data, run the *process_data.py* script in the *data* folder. This script requires the 3 arguments listed below in order to merge and clean the raw CSV files and create a SQL database.
* filepath for disaster_messages.csv
* filepath for disaster_categories.csv
* filepath including the .db file name for the output SQL database

**Model training:** To train a MultiOutputClassifier with Random Forest Classifiers, run the *train_classifier.py* script in the *models*. This script requires the following arguments:
* filepath for the SQL database created with the *process_data.py* script
* filepath including the .pkl file name to save the trained model. In case a trained model already exists in the informed filepath, the existing trained model will be loaded, without training a new model.

**Web App:** To start the web app, run the *run.py* script in the *app* folder. A web app http address will be printed in the terminal. In this Web App 2 visualizations and an input based prediction feature were added.
* Visualization 1: the count of each category in the processed database.
* Visualization 2: the top 40 most important features for each of the models trained with the MultiOutputClassifier.
* Input based prediction: a predicion applying the trained model will be performed based on the message inputed on the top of the page. The predicted categories will be highlighted in green.

## References:
[1] PYTHON. Python. Available in: https://www.python.org.

[2] SYS. sys. Available in: https://docs.python.org/3/library/sys.html

[3] PANDAS. Pandas. Available in: https://pandas.pydata.org

[4] NUMPY. NumPy. Available in: https://numpy.org

[5] SQLITE3. SQLite3. Available in: https://docs.python.org/3/library/sqlite3.html

[6] PICKLE. Pickle. Avalable in: https://docs.python.org/3/library/pickle.html

[7] NATURAL LANGUAGE TOOLKIT. Natural Language Toolkit. Available in: https://www.nltk.org

[8] SCIKIT-LEARN. Scikit-learn. Available in: https://scikit-learn.org/stable/

[9] JUPYTER NOTEBOOK. Jupyter Notebook. Available in: https://jupyter.org

[10] PLOTLY. plotly. Available in: https://plotly.com/python/

[11] JOBLIB. Joblib. Available in: https://joblib.readthedocs.io/en/latest/

[12] SQLALCHEMY. SQLAlchemy. Available in: https://www.sqlalchemy.org

[13] FLASK. Flask. Available in: https://flask.palletsprojects.com/en/2.0.x/

[14] FIGURE EIGHT. Figure Eight. Available in: https://appen.com

# Project Instructions:
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/
