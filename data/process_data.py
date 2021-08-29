import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    INPUTS:
    messages_filepath - filepath to the messages file
    categories_filepath - filepath to the categories file
    
    ACTION:
    1 - Reads the files
    2 - Merges the Dataframes from categories on messages
    
    OUTPUTS:
    df - merged Dataframe
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'left')
    return df


def clean_data(df):
    """
    INPUT:
    df - Dataframe created on the load_data function
    
    ACTIONS:
    1 - Creates a temporary categories Dataframe based on the column Category from the original df
    2 - Splits categories in separate category columns
    3 - Renames the column names according to their categories
    4 - Clears the categorical values keeping the binary value (1 or 0) as integer
    5 - Replace the original Category column with the transformed category Dataframe
    6 - Removes duplicates and NaN
    
    OUTPUTS:
    clean_df
    
    """
    # STEP 1 and 2:
    categories = df.categories.str.split(';', expand = True)
    
    # STEP 3: 
    first_row = categories[0:1].values.tolist()[0]
    category_colnames = list(map(lambda value: value[:-2], first_row))
    categories.columns = category_colnames
    
    # STEP 4: 
    for column in categories:
        categories[column] = list(map(lambda value: int(value[-1]), categories[column]))
    
    # STEP 5:
    df.drop('categories', axis = 1, inplace = True)
    clean_df = pd.concat([df, categories], axis = 1)
    
    # STEP 6:
    clean_df.drop_duplicates(inplace = True)
    clean_df.dropna(inplace = True)
    
    return clean_df


def save_data(df, database_filename):
    """
    INPUTS:
    df - Dataframe to be saved on SQL
    database_filename - SQL Database in which the Dataframe will be saved
    
    ACTIONS:
    Saves the Dataframe on the given SQL Database
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False)
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()