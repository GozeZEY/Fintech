import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# 导入数据集
arr = pd.read_excel(r".\data\train.xlsx")
test = pd.read_excel(r".\data\test_B榜.xlsx")

# ？置为缺失值
arr = arr.replace('?', np.nan)
test = test.replace('?', np.nan)

# 取得测试集CUST_UID标识
id_test = test['CUST_UID']

# 去除标签值为空的数据
arr = arr[~arr['LABEL'].isnull()]
arr = arr.reset_index(drop=True)

# 取出CUST_UID,且进行labelencoding处理
test = test.drop(columns=['CUST_UID'])
arr = arr.drop(columns=['CUST_UID'])
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

# 分割label和训练集特征
Label_train = arr['LABEL']
arr = arr.drop(columns=['LABEL'])

# 特征筛选完成，使用对抗验证对每个特征进行auc排名
test6 = test[['MON_12_EXT_SAM_TRSF_OUT_AMT', 'MON_12_EXT_SAM_NM_TRSF_OUT_CNT', 'LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV',
              'LAST_12_MON_MON_AVG_TRX_AMT_NAV', 'MON_6_50_UP_ENTR_ACT_CNT', 'MON_6_50_UP_LVE_ACT_CNT',
              'MON_12_AGV_TRX_CNT', 'MON_12_ACM_ENTR_ACT_CNT', 'MON_12_AGV_ENTR_ACT_CNT',
              'MON_12_ACM_LVE_ACT_CNT', 'SHH_BCK', 'HLD_DMS_CCY_ACT_NBR',
              'CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT', 'MON_12_CUST_CNT_PTY_ID', 'MON_12_EXT_SAM_AMT',
              'CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT', 'REG_DT', 'LGP_HLD_CARD_LVL',
              'MON_12_ACT_IN_50_UP_CNT_PTY_QTY', 'LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL',
              'CUR_YEAR_COUNTER_ENCASH_CNT', 'MON_12_ACT_OUT_50_UP_CNT_PTY_QTY',
              'LAST_12_MON_COR_DPS_DAY_AVG_BAL',
              ]]

arr6 = arr[['MON_12_EXT_SAM_TRSF_OUT_AMT', 'MON_12_EXT_SAM_NM_TRSF_OUT_CNT', 'LAST_12_MON_DIF_NM_MON_AVG_TRX_AMT_NAV',
            'LAST_12_MON_MON_AVG_TRX_AMT_NAV', 'MON_6_50_UP_ENTR_ACT_CNT', 'MON_6_50_UP_LVE_ACT_CNT',
            'MON_12_AGV_TRX_CNT', 'MON_12_ACM_ENTR_ACT_CNT', 'MON_12_AGV_ENTR_ACT_CNT',
            'MON_12_ACM_LVE_ACT_CNT', 'SHH_BCK', 'HLD_DMS_CCY_ACT_NBR',
            'CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT', 'MON_12_CUST_CNT_PTY_ID', 'MON_12_EXT_SAM_AMT',
            'CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT', 'REG_DT', 'LGP_HLD_CARD_LVL',
            'MON_12_ACT_IN_50_UP_CNT_PTY_QTY', 'LAST_12_MON_COR_DPS_TM_PNT_BAL_PEAK_VAL',
            'CUR_YEAR_COUNTER_ENCASH_CNT', 'MON_12_ACT_OUT_50_UP_CNT_PTY_QTY',
            'LAST_12_MON_COR_DPS_DAY_AVG_BAL',
            ]]

# 标签重新导入
arr6['LABEL'] = Label_train

# lgb参数设置，手动调参
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'feature_fraction': 1,
    'bagging_fraction': 0.95,
    # 训练样本随机采样
    'bagging_freq': 5,
    'num_leaves': 31,
    'max_depth': 8,
    'is_unbalance': True,
    # 正负样本不平衡
    'lambda_l1': 2,
    'lambda_l2': 0.005,
    # 正则化处理
    'verbose': -1,
    'learning_rate': 0.1,
    'random_state': 20,
    'min_data_in_leaf': 21,
    'min_sum_hessian_in_leaf': 1.2,
}

# 10折交叉
folds = KFold(n_splits=10, shuffle=True, random_state=22)

# 构造预测值序列
predictions = np.zeros(len(test))

# 开始训练
x = 1
for train_index, valid_index in folds.split(arr6):
    x = x + 1
    print()
    train_tmp = arr6.iloc[train_index]
    valid_tmp = arr6.iloc[valid_index]
    Label_train = train_tmp['LABEL']
    train_tmp = train_tmp.drop(columns=["LABEL"])
    Label_test = valid_tmp['LABEL']
    valid_tmp = valid_tmp.drop(columns=["LABEL"])
    train_data = lgb.Dataset(train_tmp, label=Label_train)
    valid_data = lgb.Dataset(valid_tmp, label=Label_test, reference=train_data)

    clf = lgb.train(params,
                    train_data,
                    num_boost_round=10000,
                    valid_sets=valid_data,
                    early_stopping_rounds=500,
                    )
    print()
    predictions += clf.predict(test6, num_iteration=clf.best_iteration) / folds.n_splits
    print(predictions)

# group = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# plt.hist(predictions, group, histtype='bar', rwidth=0.8)
# plt.show()
# sns.kdeplot(predictions, color='g')
# sns.kdeplot(predictions, color='r')
# plt.show()


# 导入测试集CUST_UID和对应的预测值进入txt文件
result = pd.DataFrame(np.zeros((len(test), 2)), columns=['id', 'label'])
result['id'] = id_test
result['label'] = predictions
result['label'] = round(result['label'], 10)
result.to_csv("test_xxx2.txt", sep=' ', index=False, header=0)
