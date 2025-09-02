from joblib import load

def main():
    # Loading trained model
    model = load("../models/sms-spam-classifier.joblib")
    vectorizer = load("../models/vectorizer.joblib")
    
    # Predicting user input SMS
    x = input("Write your sms here: ")
    
    x_vector = vectorizer.transform([x])
    
    pred = model.predict(x_vector) # Returns a NumpyArray
    print(f'The prompted sms is {pred[0]}')


if __name__ == '__main__':
    main()