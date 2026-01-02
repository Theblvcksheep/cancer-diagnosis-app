import pickle
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def create_model(data: DataFrame):
    # dataframe into predictor and target varibles
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Initalize values to normalize values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    #test model
    y_pred = model.predict(X_test)
    print(f"Accuracy of model: {accuracy_score(y_test, y_pred)}")
    print(f"Classification report: \n {classification_report(y_test, y_pred)}")

    """Accuracy of model: 0.9736842105263158
Classification report: 
               precision    recall  f1-score   support

           0       0.97      0.99      0.98        71
           1       0.98      0.95      0.96        43

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
    """

    return model, scaler

def get_clean_data():
    #import cancer data csv
    data = pd.read_csv(r"..\data\data.csv")

    # remove unnamed: 32 column
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    # transform diagmosis column to dummy varibles M = 1 and B = 0
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B':0})
    
    return data

def main():
    #clean and transform data 
    data = get_clean_data()

    #model creation
    model, scaler = create_model(data)

    # model export with pickle
    with open('model.pkl', 'wb') as f:
        pickle.dump(model,f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler,f)

if __name__ == '__main__':
    main()
