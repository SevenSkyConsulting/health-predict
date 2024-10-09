from implementation import random_forest, logistic_regression

import pandas as pd

all_data = pd.read_csv('diabetes.csv')

random_forest_accuracies = []
logistic_regression_accuracies = []

for i in range(10):
    all_data = all_data.sample(frac=1)

    train_data = all_data[10:]
    test_data = all_data[:10]

    X_train = train_data.drop(columns=['Outcome'], axis=1)
    y_train = train_data['Outcome']

    X_test = test_data.drop(columns=['Outcome'], axis=1)
    y_test = test_data['Outcome']

    rf = random_forest(X_train, y_train, X_test, y_test)
    lr = logistic_regression(X_train, y_train, X_test, y_test)

    random_forest_accuracies.append(rf)
    logistic_regression_accuracies.append(lr)

print("Random Forest: ", sum(random_forest_accuracies)/len(random_forest_accuracies))
# print("Random Forest: ", random_forest_accuracies, y_test)
# print("Random Forest: ", y_test)
print("Logistic Regression: ", sum(logistic_regression_accuracies)/len(logistic_regression_accuracies))
if __name__ == "__main__":
    pass
