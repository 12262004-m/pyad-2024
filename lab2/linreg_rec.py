import pickle
import re
import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

nltk.download("stopwords")
nltk.download("punkt")


def books_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Books.scv"""
    df = df.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"], errors="ignore")
    books['Book-Author'] = books['Book-Author'].fillna('Unknown Author')
    books['Publisher'] = books['Publisher'].fillna('Unknown Publisher')
    df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')
    df = df.dropna(subset=['Year-Of-Publication'])
    df['Year-Of-Publication'] = df['Year-Of-Publication'].astype(int)
    df = df[df['Year-Of-Publication'] <= 2024]
    return df


def ratings_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Функция для предобработки таблицы Ratings.scv
    Целевой переменной в этой задаче будет средний рейтинг книги,
    поэтому в предобработку (помимо прочего) нужно включить:
    1. Замену оценки книги пользователем на среднюю оценку книги всеми пользователями.
    2. Расчет числа оценок для каждой книги (опционально)."""

    df = df.drop(df[df['Book-Rating'] == 0].index)
    df['User-ID'] = df['User-ID'].astype('int')
    df['Book-Rating'] = df['Book-Rating'].astype('int')
    average_ratings = df.groupby("ISBN")["Book-Rating"].mean().reset_index()
    average_ratings.columns = ["ISBN", "Mean-Rating"]
    df = pd.merge(df, average_ratings, on="ISBN")
    return df


def title_preprocessing(text: str) -> str:
    """Функция для нормализации текстовых данных в стобце Book-Title:
    - токенизация
    - удаление стоп-слов
    - удаление пунктуации
    Опционально можно убрать шаги или добавить дополнительные.
    """
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)


def modeling(books: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """В этой функции нужно выполнить следующие шаги:
    1. Бинаризовать или представить в виде чисел категориальные столбцы (кроме названий)
    2. Разбить данные на тренировочную и обучающую выборки
    3. Векторизовать подвыборки и создать датафреймы из векторов (размер вектора названия в тестах – 1000)
    4. Сформировать итоговые X_train, X_test, y_train, y_test
    5. Обучить и протестировать SGDRegressor
    6. Подобрать гиперпараметры (при необходимости)
    7. Сохранить модель"""
    author_encoder = LabelEncoder()
    books['Book-Author'] = author_encoder.fit_transform(books['Book-Author'])
    publisher_encoder = LabelEncoder()
    books['Publisher'] = publisher_encoder.fit_transform(books['Publisher'])
    tfidf = TfidfVectorizer(max_features=500)
    title_vectors = tfidf.fit_transform(books['Book-Title']).toarray()

    final_books = pd.concat(
        [books[['ISBN']],
         pd.DataFrame(title_vectors, index=books.index),
         books[['Book-Author', 'Publisher', 'Year-Of-Publication']]],
        axis=1
    )
    data = pd.merge(ratings, final_books, on='ISBN')
    X = data.drop(columns=['Mean-Rating', 'ISBN'])
    y = data['Mean-Rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    scaler = StandardScaler()
    X_train = X_train.rename(str, axis="columns")
    X_test = X_test.rename(str, axis="columns")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    linreg = SGDRegressor()
    linreg.fit(X_train_scaled, y_train)
    y_pred = linreg.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    with open("linreg.pkl", "wb") as file:
        pickle.dump(linreg, file)
    return mae


if __name__ == "__main__":
    books = pd.read_csv("Books.csv", dtype={"ISBN": "str"}, low_memory=False)
    ratings = pd.read_csv("Ratings.csv", dtype={"ISBN": "str", "User-ID": "int64"})
    books = books_preprocessing(books)
    ratings = ratings_preprocessing(ratings)
    modeling(books, ratings)
