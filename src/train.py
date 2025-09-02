import pandas as pd
from sklearn.model_selection import train_test_split
from model import model
from preprocessing import preprocessing


def train():
    # Reading Dataset
    sms_dataset = pd.read_csv('../dataset/SMSSpamCollection', sep='\t', names=['Label', 'Text'])
    
    # Split data into features and labels 
    x = sms_dataset['Text']
    y = sms_dataset['Label']
    
    # Separate data into train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
   
    # Preprocessing Dataset
    x_train_vector, x_test_vector = preprocessing(x_train, x_test)
    
    # Model training and evaluation metrics
    model(x_train_vector, x_test_vector, y_train, y_test)

if __name__ == '__main__':
    train()
