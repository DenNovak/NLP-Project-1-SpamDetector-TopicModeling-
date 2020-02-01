import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv('smsspamcollection0.tsv', sep='\t')
print(len(df))

print(df.isnull().sum())
print(df['label'].value_counts())

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

text_clf = Pipeline([('countvec', CountVectorizer()),
                    ('bayes', MultinomialNB()),
                    ])
text_clf.fit(X_train, y_train)

predictions = text_clf.predict(X_test)
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.classification_report(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))

myEmail = input("Enter email to check: ")
print(text_clf.predict([myEmail])[0])
