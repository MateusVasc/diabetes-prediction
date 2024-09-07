import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class Modeling:
    def __init__(self, df):
        self.df = df

    def predict_logistic_regression(self, target, test_size=0.2):
        X_train, X_test, y_train, y_test = self.split_data(target, test_size)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        return y_test, y_pred

    def split_data(self, target, test_size):
        X = self.df.drop(target, axis=1)
        y = self.df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, y_test, y_pred):
        # accuracy_score
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        # confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

        # classification_report
        report = classification_report(y_test, y_pred)
        print("Classification Report:")
        print(report)
