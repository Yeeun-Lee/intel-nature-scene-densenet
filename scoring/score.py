import pandas as pd
import numpy as np
import sys

from sklearn.metrics import accuracy_score

def scoring(prediction, actual):

    predict = pd.read_csv(prediction, engine='python')
    actual = pd.read_csv(actual, engine='python')

    predict.id = predict.id.astype('int')
    predict = predict.sort_values(by = ['id'], axis=0)
    predict = predict.reset_index(drop=True)

    actual.id = actual.id.astype('int')
    actual = actual.sort_values(by = ['id'], axis=0)
    actual = actual.reset_index(drop=True)

    if predict.id.equals(actual.id) == False:
        print('id does not match')
        sys.exit()

    label_pred = list(predict.pred_label)
    label_actual = list(actual.pred_label)

    score = accuracy_score(label_actual, label_pred)

    print(score * 100, "%")

if __name__ == "__main__":
    scoring("prediction.csv", "actual.csv")