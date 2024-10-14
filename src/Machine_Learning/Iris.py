from sklearn.datasets import load_iris  # 加载鸢尾花数据集

from sklearn.linear_model import LogisticRegression as LR  # 加载逻辑回归模型

from sklearn.model_selection import train_test_split  # 划分数据集为训练数据与测试数据

iris = load_iris()

X = iris.data

y = iris.target  # 三分类数据集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

# 逻辑回归模型的求解器为“sag”，最大迭代次数为100次，随机参数为42。

for multi_class in ('multinomial', 'ovr'):
    clf = LR(solver='sag', max_iter=100, random_state=42, multi_class=multi_class).fit(Xtrain, Ytrain)

    # 打印两种multi_class模式下的训练分数

    # %的用法，用%来代替打印的字符串中，想由变量替换的部分。%.3f表示，保留三位小数的浮点数。%s表示，字符串。

    # 字符串后的%后使用元组来容纳变量，字符串中有几个%，元组中就需要有几个变量
    print("trainingscore:%.3f(%s)" % (clf.score(Xtrain, Ytrain), multi_class))

    print("testingscore:%.3f(%s)" % (clf.score(Xtest, Ytest), multi_class))
