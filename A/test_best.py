import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
import featuretools as ft

# 导入数据集
arr = pd.read_excel(r".\data\train.xlsx")
test = pd.read_excel(r".\data\test_A榜.xlsx")

# ？置为缺失值
arr = arr.replace('?', np.nan)
test = test.replace('?', np.nan)

# 取出测试集CUST_UID标识
id_test = test['CUST_UID']

# 去除标签值为空的数据
arr = arr[~arr['LABEL'].isnull()]
arr = arr.reset_index(drop=True)

# 手动构造特征
arr['2'] = arr['MON_12_ACM_ENTR_ACT_CNT'] - arr['MON_12_ACM_LVE_ACT_CNT']
arr['3'] = arr['MON_12_AGV_ENTR_ACT_CNT'] - arr['MON_12_AGV_LVE_ACT_CNT']

test['2'] = test['MON_12_ACM_ENTR_ACT_CNT'] - test['MON_12_ACM_LVE_ACT_CNT']
test['3'] = test['MON_12_AGV_ENTR_ACT_CNT'] - test['MON_12_AGV_LVE_ACT_CNT']

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

# 从构造的特征中取得用rf得到的特征重要性排行前二百的特征：
x = 0
with open('importance.txt') as f:
    for line1 in f:
        line1 = line1.replace('\n', '')
        if x == 200:
            break
        arr[line1] = arr3[line1].copy()
        x = x + 1
arr = arr.drop(columns=['index'])
print(arr)

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

# 从构造的特征中取得用rf得到的特征重要性排行前二百的特征：
x = 0
with open('importance.txt') as f:
    for line1 in f:
        line1 = line1.replace('\n', '')
        if x == 200:
            break
        test[line1] = test3[line1].copy()
        x = x + 1
test = test.drop(columns=['index'])
print(test)

arr['LABEL'] = Label_train


# lgb参数手动调参
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
    # 正则化处理
    'verbose': -1,
    'learning_rate': 0.1,
    'random_state': 20,
    'min_data_in_leaf': 21,
    'min_sum_hessian_in_leaf': 1.2,
}

#xgb构造对应测试数据集
test2 = xgb.DMatrix(test)

# 10折处理
folds = KFold(n_splits=10, shuffle=True, random_state=22)

# 构造对应的预测值序列
predictions = np.zeros(len(test))
predictions_xgb = np.zeros(len(test))
predictions_lgb_2 = np.zeros(len(test))

# lgb开始训练
x = 1
for train_index, valid_index in folds.split(arr):
    print("fold n°{}".format(x))
    x = x + 1
    print()
    train_tmp = arr.iloc[train_index]
    valid_tmp = arr.iloc[valid_index]
    Label_train = train_tmp['LABEL']
    train_tmp = train_tmp.drop(columns=["LABEL"])
    Label_test = valid_tmp['LABEL']
    valid_tmp = valid_tmp.drop(columns=["LABEL"])
    train_data = lgb.Dataset(train_tmp, label=Label_train)
    valid_data = lgb.Dataset(valid_tmp, label=Label_test, reference=train_data)
    # 构造好训练用的训练集和验证集
    clf = lgb.train(params,
                    train_data,
                    num_boost_round=10000,
                    valid_sets=valid_data,
                    early_stopping_rounds=100,
                    # 设置早停
                    )
    predictions += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits
    # 十折取平均


# xgb参数，部分参照lgb，其他手动调参
params_2 = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eta': 0.1,
    'max_depth': 20,
    'subsample': 0.9,
    'min_child_weight': 3,
    'colsample_bytree': 1,
    'scale_pos_weight': 3,
    'eval_metric': 'auc',
    'gamma': 0.25,
    'alpha': 2,
    'lambda': 0.005,
    'random_state': 20,
    'verbosity': 3,
}

# 同样10折处理
folds = KFold(n_splits=10, shuffle=True, random_state=22)

# 开始训练
x = 1
for train_index, valid_index in folds.split(arr):
    print("fold n°{}".format(x))
    x = x + 1
    print()
    train_tmp = arr.iloc[train_index]
    valid_tmp = arr.iloc[valid_index]
    Label_train = train_tmp['LABEL']
    train_tmp = train_tmp.drop(columns=['LABEL'])
    Label_test = valid_tmp['LABEL']
    valid_tmp = valid_tmp.drop(columns=['LABEL'])
    print(Label_train)
    train_data = xgb.DMatrix(train_tmp, label=Label_train)
    valid_data = xgb.DMatrix(valid_tmp, label=Label_test)
    # 构造好训练用的训练集和验证集
    clf_2 = xgb.train(params_2,
                      train_data,
                      num_boost_round=10000,
                      early_stopping_rounds=100,
                      # 设置早停
                      evals=[(train_data, 'train'), (valid_data, 'val')],
                      verbose_eval=1,
                      # 显示相关信息
                      )
    predictions_xgb += clf_2.predict(test2, ntree_limit=clf_2.best_ntree_limit) / folds.n_splits
    # 十折取平均


print(predictions)
print(predictions_xgb)

# 平均融合xgb和lgb的预测值
predictions = (predictions + predictions_xgb) / 2
print(predictions)

# 将测试集CUST_UID标识和对应预测值导入txt文件
result = pd.DataFrame(np.zeros((len(test), 2)), columns=['id', 'label'])
result['id'] = id_test
result['label'] = predictions
result['label'] = round(result['label'], 10)
result.to_csv("test_xxx.txt", sep=' ', index=False, header=0)
