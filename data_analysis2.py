
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import missingno as msno
from sklearn.impute import SimpleImputer
import torch
from torch import nn
from d2l import torch as d2l



#预处理
# 读取数据
data_train1 = pd.read_csv(
    "D:\\CHROME DOWNLOAD\\hs\\train.csv")
data_test1 = pd.read_csv(
    "D:\\CHROME DOWNLOAD\\hs\\test.csv")


data_train1.head()
data_train1.shape
data_train1.info()
#训练数据预处理
#训练数据类型
dtype = data_train1.dtypes
dtype.value_counts()

#训练数据缺失统计
data_train1.isnull().sum().sort_values(ascending  = False).head(20)

#训练数据缺失值可视化
#msno.matrix(data_train)

#缺失值相关性分析
#msno.heatmap(data_train)

#训练标签
#sns.displot(data_train['SalePrice'])

#评分与价格关系分析

#comment = pd.concat([data_train['SalePrice'], data_train['OverallQual']], axis = 1)
#plt.figure(figsize = (10, 8))
#sns.boxplot(y='SalePrice', x= 'OverallQual', data = comment)
#价格与建造年份的关系分析
year_pricere = pd.concat([data_train1['SalePrice'], data_train1['YearBuilt']], axis=1)
plt.figure(figsize = (20, 18))
sns.boxplot(x = 'YearBuilt', y = 'SalePrice', data = year_pricere)

#价格与售出年份的关系
Yrsold_price = pd.concat([data_train1['YrSold'],data_train1['SalePrice']], axis = 1)
plt.figure(figsize = (30, 27))
Yrsold_price.plot.scatter(x='YrSold',y = 'SalePrice', s = 4, color = 'blue')

# 绘制和价格相关的特征的散点图
TotalBsmtSF_SalePrice = pd.concat([data_train1['SalePrice'],data_train1['TotalBsmtSF']],axis=1)
plt.figure(figsize=(18,16))
TotalBsmtSF_SalePrice.plot.scatter(x='TotalBsmtSF',y='SalePrice',s=4,c='red')
#对具体缺失的列进行确认
nan_col1= data_train1.isnull().all()

nan_col2 = data_train1.isnull().any()

a = data_train1.isnull()
#移除数据过少的列
data_train2 = data_train1.dropna(axis = 1,thresh = 1120)
msno.heatmap(data_train2)
#利用均值补全数据
data_train = data_train2.fillna(method='ffill') 
#补全后的缺失值关系哦统计(当然不存在)
msno.heatmap(data_train)
#测试数据预处理
#测试数据的大小
data_test1.shape
#测试数据缺失值统计
data_test1.isnull().sum().sort_values(ascending = False).head(10000)

#缺失值可视化
#msno.matrix(data_test)

#缺失值之间的关系
msno.heatmap(data_test1)
#删除数据过少的列
data_train2 = data_train1.dropna(axis = 1,thresh = 1120)

#利用均值补全数据
data_train = data_train2.fillna(method='ffill') 

#移除数据过少的列
data_test2 = data_test1.dropna(axis = 1,thresh = 1120)

#利用均值补全数据
data_test = data_test2.fillna(method='ffill') 
#补全后的缺失值关系哦统计(当然不存在)




combined_data = pd.concat([data_train1, data_test1], axis=0)
combined_data1 = pd.concat([data_train,data_test],axis = 0)
#训练数据与测试数据对比
#非值型数据对比
numerical_features = [col for col in data_train.columns if data_train[col].dtypes != 'O']
discrete_features = [col for col in numerical_features if len(data_train[col].unique()) < 25 and col not in ['Id']]
continuous_features = [feature for feature in numerical_features if feature not in discrete_features+['Id']]
categorical_features = [col for col in data_train.columns if data_train[col].dtype == 'O']

print("Total Number of Numerical Columns : ",len(numerical_features))
print("Number of discrete features : ",len(discrete_features))
print("No of continuous features are : ", len(continuous_features))
print("Number of non-numeric features : ",len(categorical_features))

#打上数据标签
combined_data['Label']='test'
combined_data['Label'][:1460]='Train'

#对比离散数据



f,axes = plt.subplots(3,6,figsize=(30,10),sharex=False)
for i,feature in enumerate(discrete_features):
    sns.histplot(data=combined_data,x=feature,hue='Label',ax=axes[i%3,i//3])

#对比连续数据


f,axes = plt.subplots(4,6,figsize=(30,15),sharex=False)
for i,feature in enumerate(continuous_features):
    sns.histplot(data=combined_data,x=feature,hue='Label',ax=axes[i%4,i//4])
  
  

# 检查数值数据的线性分布

f,axes = plt.subplots(7,6,figsize=(50,60),sharex=False)
for i,feature in enumerate(numerical_features):
    sns.scatterplot(data=combined_data,x=feature,y="SalePrice",ax=axes[i%7,i//7])


 # 对比非数值型数据对比分析
f,axes = plt.subplots(7,7,figsize=(40,40),sharex=False)
for i,feature in enumerate(categorical_features):
    sns.countplot(data=combined_data,x=feature,hue="Label",ax=axes[i%7,i//7])
   

#时序特征分析
year_feature = [col for col in combined_data.columns if "Yr" in col or 'Year' in col]
# 然后检查一下这些特征与销售价格是否有关系
combined_data1.groupby('YrSold')['SalePrice'].median().plot() # groupby().median()表示取每一组的中位数
plt.xlabel('Year Sold')
plt.ylabel('House Price')
plt.title('House price - YearSold' )

# 绘制其他三个特征与销售价格的散点对应图

for feature in year_feature:
    if feature != 'YrSold':
        hs = combined_data1.copy()
        plt.scatter(hs[feature],hs['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()

#开始训练

# 训练数据数据预处理

all_features = pd.concat((data_train.iloc[:,:], data_test.iloc[:, :]))

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index


#对数值列进行归一化操作（涉及非值型数据的操作）
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

# 处理离散值

all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features.astype(float)
#转化为张量进行计算
#  从pandas格式中提取NumPy格式，并将其转换为张量表示用于训练。
n_train = data_train.shape[0]

train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    data_train.SalePrice.values.reshape(-1, 1), dtype=torch.float32)


#训练开始
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
# 单层线性回归模型
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net
# 计算对数均方根误差
def log_rmse(net, features, labels):
#     为了在取对数时进一步稳定该值， 将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                          torch.log(labels)))
#     以python标量的形式返回
    return rmse.item()
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
#           优化器更新参数
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
# k折交叉验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
#   每折的大小
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
            #返回训练集和验证集
    return X_train, y_train, X_valid, y_valid
# 在折交叉验证中训练k次后，返回训练和验证误差的平均值
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            torch.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

# 模型选择
k, num_epochs, lr, weight_decay, batch_size = 5, 50, 35, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

















