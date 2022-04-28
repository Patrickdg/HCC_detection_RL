import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(train_pool, val_pool=None):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_pool.drop(['id', 'class'], axis=1), train_pool['class'])
    if val_pool is not None: 
        y_preds = clf.predict(val_pool.drop(['id', 'class'], axis=1))
        print(accuracy_score(val_pool['class'], y_preds))
    return clf

if __name__ == "__main__":
    train, test = train_test_split(pd.read_csv('data/subjects.csv'), test_size=0.5)
    train, val = train_test_split(train, test_size=0.2)
    log_model = train_model(train, val)