# 6.2.2.第1组溢流特征参数预警模型
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
# from imblearn.over_sampling import SMOTE
from sklearn.externals import joblib
import matplotlib.pyplot as plt
'''
第1组溢流特征参数预警模型训练代码
训练过程：1、读取溢流第一组特征参数
          2、构建溢流预警模型
          3、模型训练及保存
输入：溢流第一组特征参数'ROPA', 'SPPA', 'HKLA'
输出：1、溢流预警模型，模型损失值、准确率
      2、模型训练过程中训练集和测试集在损失值、准确率的比较图
'''
# 读取溢流数据
data = pd.read_excel(u"PL19-3-C52ST01_minMax.xlsx", encoding="GB2312")
# 溢流预警模型的第一组特征参数
# df = data.loc[:, ['ROPA', 'SPPA', 'HKLA', 'OverFlow1']]
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
x_train, x_test, y_train, y_test = train_test_split(df, label, test_size=0.2, random_state=1234,stratify=label)
# 构建序列模型
model = Sequential()
# 三层神经网络，第一个隐含层节点数120，激活函数relu。输入维度为3，这肯定的，因为输入节点是三个
# 要用几层中间节点？==自己试试看几层效果好就用几层（我中期先用一层吧，按照handbook里面有讲最佳层数和每层的节点个数）
# #############################为什么不定义输入层？==不用专门定义输入层，在第一层隐藏节点那定义输入维度就行。如果一次分析6个特征的话，这里的input要改成6
model.add(Dense(15, activation='relu', input_dim=6))
# 第二个隐含层节点数15，激活函数relu。输出维度是15
# 经验公式推导出的节点的中心数目约为8；
model.add(Dense(15, activation='relu'))  # 本想先暂时删掉一层隐藏节点
# 输出层1个节点，激活函数sigmoid.输出维度是1
model.add(Dense(1, activation='sigmoid'))
# 模型编译
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 模型拟合，划分验证集0.25。
# 所以不是直接从原始数据划分验证集，而是在拟合的时候从训练集中划分验证集
# model fit的函数解析：https://blog.csdn.net/a1111h/article/details/82148497
# 训练轮数越高越好，但是我先设置成1，这样后面还能有提高的空间。
# fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况
# history = model.fit(x_train, y_train, validation_split=0.25, epochs=150, batch_size=128, verbose= 1)
"""
epoch          acc
5              0.88
50             0.95     
"""
# 原epoch是50
history = model.fit(x_train, y_train, validation_split=0.25, epochs=200, batch_size=128, verbose=1)

# 保存模型
# 参考链接：https://blog.csdn.net/Andrew_jdw/article/details/82656605
model.save('model1.h5')
# 利用测试集判断模型的损失值和准确率
# in the model.compule before. we set the type of accuracy and we use acc as the metrics.
# Thus, evaluate will return "binary_crossentropy" and "accuracy"
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print("loss:", loss)
print("acc:", acc)
# print("the score: ", model.score(x_test, y_test)) !score is for sklearn instead of keras. no usage here
# 模型训练过程中训练集和验证集相同迭代次数下准确率比较可视化。plt是上面已经引入的包
plt.figure(figsize=[20, 5])
# subplot：12表示大图是一行两列两个图像，最后一个1表示当前小图像在大图的第一个位置
plt.subplot(121)
# his.his的功能在上面说了
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='validation')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
# 展现了准确率随着迭代次数增加的规律
plt.xlabel('Epoch')
plt.legend(loc=2)
# 模型训练过程中训练集和验证集相同迭代次数下损失值比较可视化
plt.subplot(122)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Model loss')
plt.ylabel('Loss')
# 展现了loss随着迭代次数增加的规律
plt.xlabel('Epoch')
plt.legend(loc=1)
plt.show()


