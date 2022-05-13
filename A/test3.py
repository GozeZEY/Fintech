from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
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

# 丢弃一些特征重要性低的原始特征
arr = arr.drop(
    columns=['MON_12_EXT_SAM_TRSF_OUT_AMT', 'MON_12_EXT_SAM_NM_TRSF_OUT_CNT', 'CUR_MON_EXT_SAM_CUST_TRSF_IN_AMT',
             'CUR_MON_EXT_SAM_CUST_TRSF_OUT_AMT', 'SHH_BCK', 'COUNTER_CUR_YEAR_CNT_AMT', 'HLD_FGN_CCY_ACT_NBR',
             'CUR_YEAR_COUNTER_ENCASH_CNT',
             'WTHR_OPN_ONL_ICO', 'MON_12_CUST_CNT_PTY_ID'])

# 分隔标签和特征
Label_train = arr['LABEL']
arr = arr.drop(columns=['LABEL'])
arr = arr.drop(columns=['CUST_UID'])

# labelencoding处理
arr = arr.replace('A', 0)
arr = arr.replace('B', 1)
arr = arr.replace('C', 2)
arr = arr.replace('D', 3)
arr = arr.replace('E', 4)
arr = arr.replace('F', 5)
arr = arr.replace('Y', 6)
print(arr)

# ft自动构造特征
es = ft.EntitySet(id='single_dataframe')  # 用id标识实体集
# 增加一个数据框，命名为iris
es.entity_from_dataframe(entity_id='iris',
                         dataframe=arr,
                         index='index',
                         make_index=True)

trans_primitives = ['add_numeric', 'subtract_numeric']  # 2列相加减来生成新特征
# ft.list_primitives()  # 查看可使用的特征集元
arr, feature_names = ft.dfs(entityset=es,
                            target_entity='iris',
                            max_depth=1,  # max_depth=1，只在原特征上进行运算产生新特征
                            verbose=1,
                            trans_primitives=trans_primitives
                            )
arr = round(arr, 6)
print(arr)
feat_labels = arr.columns

rf = RandomForestRegressor(n_estimators=100, max_depth=None)
rf_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('standardize', StandardScaler()), ('rf', rf)])
rf_pipe.fit(arr, Label_train)

# 根据随机森林模型的拟合结果选择特征
rf = rf_pipe.__getitem__('rf')
importance = rf.feature_importances_
print(importance)

# np.argsort()返回待排序集合从下到大的索引值，[::-1]实现倒序，即最终imp_result内保存的是从大到小的索引值
imp_result = np.argsort(importance)[::-1][:]

# 按重要性从高到低输出属性列名和其重要性
for i in range(len(imp_result)):
    print("%2d. %-*s %f" % (i + 1, 30, feat_labels[imp_result[i]], importance[imp_result[i]]))

# 将排好序的特征存入importance.txt中便于取用
with open('importance.txt', 'w') as f:
    for i in range(len(imp_result)):
        f.write(feat_labels[imp_result[i]])
        f.write('\n')
