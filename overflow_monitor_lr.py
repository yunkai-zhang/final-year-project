
"""
import numpy as np

class LogisticRegrassionUsingGradientDescent(object):

    # This is a Logistic Regression Classifier which uses gradient descent


    def __init__ (self, learning_rate=0.05, num_epochs=100, randon_state=1):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.randon_state = randon_state

    def fit (self, X, y):
        # initialize weights
        random_generator = np.random.RandomState (self.randon_state)
        self.weights = random_generator.normal (loc = 0, scale = 0.01, size = 1+X.shape[1])
        # initialize the list for recording costs in each epoch
        self.costs_epochs=[]

        for i in range (self.num_epochs):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            # modify weights according to errors after each epoch
            # X.T.dot(errors) means transpose first and than dot multiply the errors using mastrix product(矩阵内积）
            self.weights[1:] += self.learning_rate * X.T.dot(errors)
            self.weights[0] += self.learning_rate * errors.sum()

            cost = (-y.dot(np.log(output)) - ((1-y).dot(np.log(1 - output))))
            # giving value to costs_epochs after each epoch
            self.costs_epochs.append(cost)
        return self

    # conbine weights and input X
    def net_input(self,X):
        return np.dot(X, self.weights[1:])+self.weights[0]

    # make activation function
    def activation (self, z):
        return 1./(1. + np.exp(-np.clip(z, -250, 250)))

    #
    def predict(self,X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 读取溢流数据
data = pd.read_excel(u"PL19-3-C52ST01_minMax.xlsx", encoding="GB2312")
# 溢流预警模型的第一组特征参数
df = data.loc[:, ['Time', 'ROPA', 'SPPA', 'HKLA', 'TVA', 'GASA', 'MFOP', 'OverFlow']]
label = df['OverFlow']
# 当你要删除某一行或者某一列时，用drop函数，它不改变原有的df中的数据，而是返回另一个dataframe来存放删除后的数据。删除列要加axis=1，默认是删除行的。
df = df.drop("OverFlow", axis=1)
df = df.drop("Time", axis=1)
# 数据转换成数组格式
df = np.array(df)
# 划分训练集0.8，测试集0.2
# test split函数介绍https://www.cnblogs.com/bonelee/p/8036024.html
# 为什么这里划分的时候不用validation集呢？==貌似cv集只在逻辑回归模型里面有==不对，神经网络也有cv集，但是现在暂时把cv集合train集不分开，等到后面再分开
# randon-state为不同实数时，train_test_split抓取不同的train集合。实数一定时，抓取的train集合一定。不设置randon-state时（none），每次运行时抓取的train集都不一样。
x_train, x_test, y_train, y_test = train_test_split(df, label, test_size=0.2, random_state=1234)

# 调用模型，但是并未经过任何调参操作，使用默认值
lr_model = LogisticRegression(C = 100, random_state = 1234)
 # 训练模型
lr_model.fit(x_train, y_train)
# 获取测试集的评分
# Return the mean accuracy on the given test data and labels.
# (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)->methods
print(lr_model.score(x_test, y_test))

"""
回归模型参数                                                                正确率
默认                                                                       0.8702810105024127
C = 100, random_state = 1234                                              0.8707540921563062
"""




