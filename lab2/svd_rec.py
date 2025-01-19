import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(df[df['Book-Rating'] == 0].index)
    df['User-ID'] = df['User-ID'].astype('int')
    df['Book-Rating'] = df['Book-Rating'].astype('int')
    user_encoder = LabelEncoder()
    df['User-ID'] = user_encoder.fit_transform(df['User-ID'])
    return df


def modeling(ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Разбить данные на тренировочную и обучающую выборки
    2. Обучить и протестировать SVD
    3. Подобрать гиперпараметры (при необходимости)
    4. Сохранить модель"""

    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)
    svd = SVD()
    svd.fit(trainset)
    predictions_svd = svd.test(testset)
    mae = accuracy.mae(predictions_svd)
    with open("svd.pkl", "wb") as file:
        pickle.dump(svd, file)
    return mae


if __name__ == "__main__":
    ratings_df = pd.read_csv("ratings.csv")
    ratings_df = ratings_preprocessing(ratings_df)
    modeling(ratings_df)