import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import joblib
import warnings

data = pd.read_csv('../../data/boston_house_price.csv')


def ignore_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return wrapper


l_train = len(data[data['MEDV'].notnull()])
train = data[:l_train]
y = train['MEDV']
x = train.drop('MEDV', axis=1).values


def scoring(model):
    r = cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=5)
    score = -r
    return score


clf = Lasso(alpha=0.0005)
score = scoring(clf)
print("偏差：{:.4f}({:.4f})".format(score.mean(), score.std()))
clf.fit(x, y)
joblib.dump(clf, '../../models/boston_lasso.pkl')
print("特征总数：%d" % len(data.columns))
print("嵌入式选择后，保留特征数：%d" % np.sum(clf.coef_ != 0))


def plot_learning_curve(estimator, title, x, y, cv=10, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, cv=cv,
                                                            n_jobs=n_jobs, scoring='neg_mean_squared_error',
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="g")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="r")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="g",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt


clf = Lasso(alpha=0.0005)
plot_learning_curve(clf, "Lasso", x, y)
plt.show()
joblib.dump(clf, '../../models/boston_lasso.pkl')
print("模型保存成功！")

# clf = joblib.load('../models/boston_lasso.pkl')
# test = train[l_train:].drop('MEDV', axis=1)
# predict = np.exp(clf.predict(test))
# resul = pd.DataFrame()
# resul['MEDV'] = predict
# resul.to_csv('../result/boston_house_price_predict.csv', index=False)
