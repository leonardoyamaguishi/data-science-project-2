# Disaster Response Pipeline Project

## About this project:
This is the second project from Udacity's Data Science Nanodegree. The objective is to develop an Extract Transform and Load pipeline and a Machine Learning pipeline for NLP.

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

# Original Project Instructions:
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
