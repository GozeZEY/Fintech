

模型为lgb单模，超参数为手动调参，使用10折交叉；

test_search.py文件用于对原始特征按照两个一组进行对抗验证，以auc作为评价指标对不同特征在测试集和训练集上数据分布的差距做出量化评价，执行后生成importance2.txt文件，auc分数从低到高（auc越低，说明分布差距越小）；

test_best_b.py文件:执行后生成test_xxx_2.txt文件,用于提交验证；

（文件中的importance.txt文件来源于A榜中生成的importance.txt文件，用于test_search.py文件）
