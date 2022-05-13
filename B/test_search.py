import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

import featuretools as ft

arr = pd.read_excel(r".\data\train.xlsx")
test = pd.read_excel(r".\data\test_B榜.xlsx")

# ？置为缺失值
arr = arr.replace('?', np.nan)
test = test.replace('?', np.nan)
id_test = test['CUST_UID']
# 去除标签值为空的数据
arr = arr[~arr['LABEL'].isnull()]
arr = arr.reset_index(drop=True)

# 分割标签值和特征
test = test.drop(columns=['CUST_UID'])
arr = arr.drop(columns=['CUST_UID'])

# 进行labelencoding处理
arr = arr.replace('A', 0)
arr = arr.replace('B', 1)
arr = arr.replace('C', 2)
arr = arr.replace('D', 3)
arr = arr.replace('E', 4)
arr = arr.replace('F', 5)
arr = arr.replace('Y', 6)

test = test.replace('A', 0)
test = test.replace('B', 1)
test = test.replace('C', 2)
test = test.replace('D', 3)
test = test.replace('E', 4)
test = test.replace('F', 5)
test = test.replace('Y', 6)

# 分隔训练集标签和特征
Label_train = arr['LABEL']
arr = arr.drop(columns=['LABEL'])

# ft自动构造特征
es = ft.EntitySet(id='single_dataframe')  # 用id标识实体集
# 增加一个数据框，命名为iris
es.entity_from_dataframe(entity_id='iris',
                         dataframe=arr,
                         index='index',
                         make_index=True)

trans_primitives = ['add_numeric', 'subtract_numeric']  # 2列相加减乘除来生成新特征
# ft.list_primitives()  # 查看可使用的特征集元
arr3, feature_names = ft.dfs(entityset=es,
                             target_entity='iris',
                             max_depth=1,  # max_depth=1，只在原特征上进行运算产生新特征
                             verbose=1,
                             trans_primitives=trans_primitives
                             )
arr3 = round(arr3, 6)

# 取出一个自动构造的特征重要性参与原始特征对抗验证排行
x = 0
with open('importance.txt') as f:
    for line1 in f:
        line1 = line1.replace('\n', '')
        if x == 1:
            break
        arr[line1] = arr3[line1].copy()
        x = x + 1
arr = arr.drop(columns=['index'])

es_test = ft.EntitySet(id='single_dataframe_test')  # 用id标识实体集
# 增加一个数据框，命名为iris_test
es_test.entity_from_dataframe(entity_id='iris_test',
                              dataframe=test,
                              index='index',
                              make_index=True)

trans_primitives = ['add_numeric', 'subtract_numeric']  # 2列相加减来生成新特征
# ft.list_primitives()  # 查看可使用的特征集元
test3, feature_names_test = ft.dfs(entityset=es_test,
                                   target_entity='iris_test',
                                   max_depth=1,  # max_depth=1，只在原特征上进行运算产生新特征
                                   verbose=1,
                                   trans_primitives=trans_primitives
                                   )
test3 = round(test3, 6)

# 取出一个自动构造的特征重要性参与原始特征对抗验证排行
x = 0
with open('importance.txt') as f:
    for line1 in f:
        line1 = line1.replace('\n', '')
        if x == 1:
            break;
        test[line1] = test3[line1].copy()
        x = x + 1
test = test.drop(columns=['index'])

# 对抗验证：构造是否为测试集的标签，分别对每组特征子集进行auc评分
test['is_test'] = 1
arr['is_test'] = 0

# 合并训练集和测试集
test5 = pd.concat([arr, test])
test5 = test5.reset_index(drop=True)

# 分隔出标签
Label_test_2 = test5['is_test']
test5 = test5.drop(columns=['is_test'])

# 构造分数集合（原始特征49个，自动构造特征1个，两个一组，一个25组分数）
score = np.zeros(25)

# 使用a榜时的参数，使用lgb进行auc评分
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'feature_fraction': 1,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'num_leaves': 31,
    'max_depth': 20,
    'is_unbalance': True,
    'lambda_l1': 2,
    'lambda_l2': 0.005,
    'verbose': -1,
    'learning_rate': 0.1,
    'random_state': 20,
    'min_data_in_leaf': 21,
    'min_sum_hessian_in_leaf': 1.2,
}
print(test5)
print(Label_test_2)

# 对某些区间进行数量统计
x = 0
y = 0
z = 0

# 开始训练
for i in range(0, 25):
    print(i)
    print(test5.columns[2 * i])
    x_train, x_test, y_train, y_test = train_test_split(test5[[test5.columns[2 * i], test5.columns[2 * i + 1]]],
                                                        Label_test_2, train_size=0.8,
                                                        random_state=22)


    train_data = lgb.Dataset(x_train, label=y_train)
    valid_data = lgb.Dataset(x_test, label=y_test)
    clf = lgb.train(params,
                    train_data,
                    num_boost_round=10000,
                    valid_sets=valid_data,
                    early_stopping_rounds=500,
                    )
    predictions = clf.predict(x_test, num_iteration=clf.best_iteration)
    score[x] = roc_auc_score(y_test, predictions)
    print(predictions)
    print(score[x])
    if score[x] <= 0.6:
        y = y + 1
    if score[x] >= 0.7:
        z = z + 1
    x = x + 1

print(y)
print(z)

# 对分数进行排序，并返回对应的索引
imp_result = np.argsort(score)
print(imp_result)


# 打印分数排序
for i in range(0, len(imp_result)):
    print("%2d. %-*s %-*s %f" % (
    i + 1, 30, test5.columns[2 * imp_result[i]], 30, test5.columns[2 * imp_result[i] + 1], score[imp_result[i]]))

# 将排好序的特征存入importance2.txt文件，便于观察和选用特征
with open('importance2.txt', 'w') as f:
    for i in range(len(imp_result)):
        f.write(test5.columns[2 * imp_result[i]])
        f.write('\n')
        f.write(test5.columns[2 * imp_result[i] + 1])
        f.write('\n')

