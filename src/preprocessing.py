from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocessing(x_train, x_test):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1,2)
    )
    x_train_vector = vectorizer.fit_transform(x_train)
    x_test_vector = vectorizer.transform(x_test)
    
    dump(vectorizer, '../models/vectorizer.joblib')
    return x_train_vector, x_test_vector