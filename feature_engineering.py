
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(text_series):
    vectorizer = TfidfVectorizer(max_features=500)
    X_text = vectorizer.fit_transform(text_series.fillna(""))
    return X_text, vectorizer
