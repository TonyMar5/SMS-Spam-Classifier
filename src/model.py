from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump

def model(x_train_vector, x_test_vector, y_train, y_test):
    
    model = LogisticRegression()
    model.fit(x_train_vector, y_train)

    # Evaluate model on testing set
    y_pred = model.predict(x_test_vector)
    metrics = classification_report(y_test, y_pred)
    print(metrics)
    
    dump(model, '../models/sms-spam-classifier.joblib')