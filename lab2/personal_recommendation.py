import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import SGDRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from surprise import SVD, Dataset, Reader


def data_preprocessing(books, ratings):
    books = books.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"], errors="ignore")
    books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors="coerce")
    books['Book-Author'] = books['Book-Author'].fillna('Unknown Author')
    books['Publisher'] = books['Publisher'].fillna('Unknown Publisher')
    books["Book-Title-Code"] = books["Book-Title"].factorize()[0]
    imputer = SimpleImputer(strategy='median')
    numeric_cols = books.select_dtypes(include=[np.number]).columns
    books[numeric_cols] = imputer.fit_transform(books[numeric_cols])
    return books, ratings


def get_zero_user(ratings):
    zero_ratings = ratings[ratings['Book-Rating'] == 0]
    user = zero_ratings['User-ID'].value_counts().idxmax()
    return user


def svd_modeling(ratings):
    with open('svd.pkl', 'rb') as f:
        svd = pickle.load(f)

    zero_ratings = ratings[ratings['Book-Rating'] == 0]
    predictions = []
    for ind, row in zero_ratings.iterrows():
        prediction = svd.predict(row['User-ID'], row['ISBN'])
        predictions.append((row['ISBN'], prediction.est))

    predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)
    recommended_books = [book for book, _ in predictions_sorted if _ >= 8]
    return recommended_books


def linreg_modeling(books):
    with open('linreg.pkl', 'rb') as f:
        linreg = pickle.load(f)
    X = books.drop(['ISBN'], axis=1).select_dtypes(include=[np.number])
    predicted_ratings = linreg.predict(X)
    recommended_books = books['ISBN'][predicted_ratings >= 8].tolist()
    return recommended_books, linreg


books_df = pd.read_csv('Books.csv')
ratings_df = pd.read_csv('Ratings.csv')
books_df, ratings_df = data_preprocessing(books_df, ratings_df)
user_id = get_zero_user(ratings_df)
recommended_books_svd = svd_modeling(ratings_df)
recommended_books_linreg, linreg_model = linreg_modeling(books_df)
recommended_books_sorted = sorted(recommended_books_linreg, key=lambda book: linreg_model.predict(
    books_df[books_df['ISBN'] == book].drop(['ISBN'], axis=1).select_dtypes(include=[np.number])), reverse=True)