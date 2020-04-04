# data preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model


# from sklearn.externals import joblib

'''
Codes for prediction
Process: 1. Set the x and y 
         2. Get pred-y and compare it with y

input: data of a new well
output: the prediction
'''
# read the data
# Encoding shows how characters are coded.
# Though we normally use utf8, we use gb2312 here to represent Chinese characters
data = pd.read_excel(u"PL19-3-C40ST01(for prediction).xlsx", encoding="GB2312")
# select features of overflow
# It means we will extract following columns：'Time', 'ROPA','SPPA', 'HKLA', 'TVA', 'GASA', 'MFOP',  'OverFlow'
df = data.loc[:, ['Time', 'ROPA', 'SPPA', 'HKLA', 'TVA', 'GASA', 'MFOP', 'OverFlow', 'OverFlowPreDiction']]
# Select the data within the time 15 minutes before and after the overflow(this over flow did not last for a long
# time(15 minutes), so we do not need that much time
# pd.to_datetime(df['Time'] In df Dataframe, Change the format of the data from String to Time,
# because data in Time format instead of String format can be sorted
df = df[(pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S') >= pd.to_datetime('2016/5/6 23:45:00',
                                                                                  format='%Y-%m-%d %H:%M:%S')) &
        (pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S') <= pd.to_datetime('2016/5/7 2:00:00',
                                                                                  format='%Y-%m-%d %H:%M:%S'))]
# Set the label Overflow to 0
# Extract certain column and set its value to 0
df.loc[:, 'OverFlow'] = 0
df.loc[:, 'OverFlowPrediction'] = 0
# The label Overflow is set to 1 if there is overflow at that time
# This is set according to the overflow record in the Word file.
# For the "log" function,
# the content before "," determines the range of rows, while the content after "," determines columns
df.loc[(pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S') >= pd.to_datetime('2016/5/7 00:45:00',
                                                                                 format='%Y-%m-%d %H:%M:%S')) &
       (pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S') <= pd.to_datetime('2016/5/7 1:00:00',
                                                                                 format='%Y-%m-%d %H:%M:%S')), 'OverFlow'] = 1
# Write data with the label of overflow(Overflow) to the file
# The original data with more negative label is overwritten
df.to_excel(u"PL19-3-C40ST01(for prediction LABELED).xlsx", index=False)
# Normalize the data
Data = pd.read_excel(u"PL19-3-C40ST01(for prediction LABELED).xlsx", encoding='GB2312')
# If I want to analyse the data with two groups of three features,
# I will need two labels, Overflow and Overflow1, to match each group
# Data = Data.loc[:, ['Time', 'ROPA', 'SPPA', 'HKLA', 'TVA', 'GASA', 'MFOP', 'OverFlow', 'OverFlow1']]
df1 = Data.loc[:, 'Time']
# df2 = Data.loc[:, ['OverFlow', 'OverFlow1']]
df2 = Data.loc[:, ['OverFlow', 'OverFlowPreDiction']]
# Normalize the data to the range of [0,1].
# minMax is a normalizor which is constructed by MinMaxScaler() with default parameters.
# The subfunction minMax.fit_transform can be used for applying normalization
minMax = MinMaxScaler()
# Select following columns to do the normalization
data_minMax = minMax.fit_transform(Data.loc[:, ['ROPA', 'SPPA', 'HKLA', 'TVA', 'GASA', 'MFOP']])
# 把归一化的数据制作成数据帧，等待合并
# Convert the normalized data to the DataFrame and wait to be merged with "Time" and "Overflow"
df3 = pd.DataFrame(data_minMax, columns=['ROPA', 'SPPA', 'HKLA', 'TVA', 'GASA', 'MFOP'])
# Do alignment in row for three files, and combine them
df = pd.concat([df1, df3, df2], axis=1)
# Save the normalized data with feature "Time" and label "Overflow"
df.to_excel("PL19-3-C40ST01(for prediction LABELEDandMINMAXED).xlsx", index=False)

# read the file and be prepared to do the prediction
data = pd.read_excel(u"PL19-3-C40ST01(for prediction LABELEDandMINMAXED).xlsx", encoding="GB2312")
# select features of overflow
# It means we will extract following columns：'Time', 'ROPA','SPPA', 'HKLA', 'TVA', 'GASA', 'MFOP',  'OverFlow'
df = data.loc[:, ['Time', 'ROPA', 'SPPA', 'HKLA', 'TVA', 'GASA', 'MFOP', 'OverFlow', 'OverFlowPreDiction']]

# load the exist model
model = load_model('model1.h5')

# do the prediction
# x, x_test, y, y_test = train_test_split(df, label, test_size=0.2, random_state=1234)
df.loc[:, 'OverFlowPrediction'] = model.predict_classes(df.loc[:, ['ROPA', 'SPPA', 'HKLA', 'TVA', 'GASA', 'MFOP']],
                                                        verbose=0)
df.to_excel("PL19-3-C40ST01(for prediction PREDICTED).xlsx", index=False)