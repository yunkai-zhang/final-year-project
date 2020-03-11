# data preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# from sklearn.externals import joblib

'''
Codes for data preprocessing
Process: 1. Select the main feature of overflow
         2. Tag the data according to the time when overflow occurs
         3. Normalize the data
         
input: the overflow data of all the wells
output: the normalized data of the overflow feature
'''
# read the data
# Encoding shows how characters are coded.
# Though we normally use utf8, we use gb2312 here to represent Chinese characters
data = pd.read_excel(u"PL19-3-C52ST01.xlsx", encoding="GB2312")
# select features of overflow
# It means we will extract following columns：'Time', 'ROPA','SPPA', 'HKLA', 'TVA', 'GASA', 'MFOP',  'OverFlow'
df = data.loc[:, ['Time', 'ROPA', 'SPPA', 'HKLA', 'TVA', 'GASA', 'MFOP', 'OverFlow']]
# Select the data within the time one day before and after the overflow
# pd.to_datetime(df['Time'] In df Dataframe, Change the format of the data from String to Time,
# because data in Time format instead of String format can be sorted
df = df[(pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S') >= pd.to_datetime('2016/3/29 4:57:00',
                                                                                  format='%Y-%m-%d %H:%M:%S')) &
        (pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S') <= pd.to_datetime('2016/4/1 6:40:00',
                                                                                  format='%Y-%m-%d %H:%M:%S'))]
# Set the label Overflow to 0
# Extract certain column and set its value to 0
df.loc[:, 'OverFlow'] = 0
# The label Overflow is set to 1 if there is overflow at that time
# This is set according to the overflow record in the Word file.
# For the "log" function,
# the content before "," determines the range of rows, while the content after "," determines columns
df.loc[(pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S') >= pd.to_datetime('2016/3/30 4:57:00',
                                                                                 format='%Y-%m-%d %H:%M:%S')) &
       (pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S') <= pd.to_datetime('2016/3/31 6:40:00',
                                                                                 format='%Y-%m-%d %H:%M:%S')), 'OverFlow'] = 1
# Write data with the label of overflow(Overflow) to the file
# The original data with more negative label is overwritten
df.to_excel(u"PL19-3-C52ST01.xlsx", index=False)
# Normalize the data
Data = pd.read_excel(u"PL19-3-C52ST01.xlsx", encoding='GB2312')
# If I want to analyse the data with two groups of three features,
# I will need two labels, Overflow and Overflow1, to match each group
# Data = Data.loc[:, ['Time', 'ROPA', 'SPPA', 'HKLA', 'TVA', 'GASA', 'MFOP', 'OverFlow', 'OverFlow1']]
df1 = Data.loc[:, 'Time']
# df2 = Data.loc[:, ['OverFlow', 'OverFlow1']]
df2 = Data.loc[:, ['OverFlow']]
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
df.to_excel("PL19-3-C52ST01_minMax.xlsx", index=False)
