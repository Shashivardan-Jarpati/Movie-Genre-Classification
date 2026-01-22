from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def build_model():
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    model = MultinomialNB()
    return vectorizer, model
