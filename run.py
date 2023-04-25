import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import preproc


def train(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', svm.LinearSVC(C=1.0)),
    ])

    text_clf.fit(X_train, y_train)

    predicted = text_clf.predict(X_test)

    print(f'Accuracy: {round(float(np.mean(predicted == y_test)), 3) * 100}%')


def get_dataset(df: pd.DataFrame):
    df.dropna(inplace=True)
    df = df.loc[:, ['reviewText', 'overall']]
    df['reviewText'] = df['reviewText'].apply(preproc)
    df['overall'] = df['overall'].astype(int)
    return df.reviewText.values, df.overall.values


if __name__ == '__main__':
    json_df = pd.read_json('data/review.json', orient='records')
    train(*get_dataset(json_df))
