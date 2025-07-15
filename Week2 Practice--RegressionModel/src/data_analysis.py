
import pandas as pd

def clean_data(df):
    df = df[df['Package'].notnull()]
    df['Price'] = df['Low Price']
    df = df[['City Name', 'Package', 'Variety', 'Origin', 'Item Size', 'Price']]
    df = df.dropna()
    X = df.drop('Price', axis=1)
    y = df['Price'].tolist()
    return X, y

def split_data(X, y, test_size=0.2):
    n = int(len(X) * (1 - test_size))
    train_x = X.iloc[:n]
    train_y = y[:n]
    test_x = X.iloc[n:]
    test_y = y[n:]
    return train_x, test_x, train_y, test_y
