'''KNN算法'''
# https://www.cnblogs.com/ahu-lichang/p/7151007.html
'''
输入:     newInput:  (1xN)的待分类向量
             dataSet:   (NxM)的训练数据集
             labels:     训练数据集的类别标签向量
             k:         近邻数
输出:     可能性最大的分类标签
'''



'''myKNN.py脚本文件'''
from numpy import *
import operator

# 创建一个数据集，包含2个类别共4个样本
def createDataSet():
    # 生成一个矩阵，每行表示一个样本
    group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    # 4个样本分别所属的类别
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# KNN分类算法函数定义
def KNNClassify(newInput, dataSet, labels, k):
    numSample = dataSet.shape[0]# shape[0]表示行数
    # tile(A, reps): 构造一个矩阵，通过A重复reps次得到
    diff = tile(newInput, (numSample, 1)) - dataSet# 按元素求差值
    squaredDiff = diff ** 2# 将差值平方
    squaredDist = sum(squaredDiff, axis = 1)# 按行累加
    distance = squaredDist ** 0.5# 将差值平方和求开方，即得距离
    # 对距离排序
    sortedDistIndices = argsort(distance)# argsort() 返回排序后的索引值
    classCount = {}# 定义一个空字典
    for i in range(0, k):
        # 选择k个最近邻
        voteLabel = labels[sortedDistIndices[i]]
        # 计算k个最近邻中各类别出现的次数
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1# When the key voteLabel is not in dictionary classCount, get() will return 0
        # 返回出现次数最多的类别标签
        maxCount = 0
        for key, value in classCount.items():
            if value > maxCount:
                maxCount = value
                maxIndex = key
        return maxIndex















