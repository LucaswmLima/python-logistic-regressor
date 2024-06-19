import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

class RegressionLogic:
    def __init__(self):
        self.model = None
        self.accuracy = None
        self.confusion_matrix = None

    def load_data(self, filepath, sep):
        self.data = pd.read_csv(filepath, sep=sep)
        return self.data

    def run_regression(self, x_column, y_column):
        self.X = self.data[x_column].values.reshape(-1, 1)
        self.y = self.data[y_column].values
        self.model = LogisticRegression()
        self.model.fit(self.X, self.y)
        self.accuracy = accuracy_score(self.y, self.model.predict(self.X))
        self.confusion_matrix = confusion_matrix(self.y, self.model.predict(self.X))

    def plot_regression(self):
        plt.figure()
        plt.scatter(self.X, self.y, color='blue', label='Dados Reais')
        X_test = np.linspace(self.X.min(), self.X.max(), 300).reshape(-1, 1)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        plt.plot(X_test, y_prob, color='red', label='Previsão')
        plt.xlabel("X")
        plt.ylabel("Probabilidade de Y")
        plt.title("Regressão Logística")
        plt.legend()
        plt.show()

    def predict(self, x_value):
        x_value = np.array(x_value).reshape(-1, 1)
        return self.model.predict(x_value)
