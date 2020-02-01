import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

npr = pd.read_csv('nprdataset.csv')

tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(npr['Article'])

nmf_model = NMF(n_components=7)
nmf_model.fit(dtm)

for index, topic in enumerate(nmf_model.components_):
    print(f"Top 15 words for topic # {index}")
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]]) # words with highest coefficient values
    print('\n')

topic_results = nmf_model.fit_transform(dtm)

npr['Topic'] = topic_results.argmax(axis=1)
print(npr.head(10))
