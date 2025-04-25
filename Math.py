import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from scipy.optimize import minimize
import os
import torch
DATA_LEN =26
# 定义回归模型（需要选手自己设计填写）
# Define the regression model (participants need to design and fill it in themselves)

# 进行回归的函数（需要选手自己设计填写）
# Function for parameter solving (participants need to design and fill it in themselves)


def generateData(p,q,b0,b1,b2):
    b_seq = []
    b_seq.append(b0)
    b_seq.append(b1)
    b_seq.append(b2)
    for i in range(DATA_LEN - 3):
        b_seq.append(p*b_seq[-2] - q*b_seq[-3])
    a_seq = np.array(b_seq)
    a_seq = a_seq+0.5
    a_seq = np.floor(a_seq)
    return a_seq,b_seq


def pqregression(data):
    n = len(data) - 3
    A = []
    Y = []
    for i in range(n):
        A.append([data[i+1], data[i]])  # 系数矩阵
        Y.append(data[i+3])              # 结果向量
    A = np.array(A)
    Y = np.array(Y)
    p_q = np.linalg.lstsq(A, Y, rcond=None)[0]

    return p_q #p_value, q_value
def validate(a_seq, p, q):
    """ 验证求得的 p 和 q """
    b_seq = reconstruct_b(a_seq)
    for i in range(len(b_seq) - 3):
        b_pred = p * a_seq[i+1] - q * a_seq[i]
        if np.round(b_pred + 0.5) != a_seq[i+3]:
            print(i,b_pred, a_seq[i+3])
    print("ok")
    return True
def objective_function(params,seq):
    p, q, b0, b1, b2 = params
    t_seq,_= generateData(p,q,b0,b1,b2)
    return np.mean((t_seq - seq)**2)
def pq_scipy_regression(data):
     b0, b1, b2=data[0], data[1], data[2]
     #p,q = pqregression(data)
     result = minimize(objective_function, [1,1,b0,b1,b2],args=(data), method='Powell')
     t_seq,_ = generateData(result.x[0],result.x[1],result.x[2], result.x[3], result.x[4])
     #print(result)
     #for i in range(len(data)):
         #print(data[i],t_seq[i])

def reconstruct_b(a_seq):
    """ 重建数列 b_n 的近似值 """
    b = [ a for a in a_seq]  # 直接取中间值作为近似
    return b
def main():
    #-------------读取训练集,训练集地址已经设定好，下面这段不用修改------------------#
    #-----Read the training set, the address of the training set has been set, and the following section does not need to be modified-------#
    train_path = "D:/AI4S_Teen_Cup_2025/dataset/Math/a_seq_train.csv"   #"/bohr/train-btk3/v1/a_seq_train.csv"
    data_train = pd.read_csv(train_path)
    a_seq = [a for a in data_train['a_seq']]

    #-------------读取测试集---------------#“DATA_PATH”是测试集加密后的环境变量，按照如下方式可以在提交后，系统评分时访问测试集，但是选手无法直接下载
    #----Read the testing set, “DATA_PATH” is an environment variable for the encrypted test set. After submission, you can access the test set for system scoring in the following manner, but the contestant cannot download it directly.-----#
    if os.environ.get('DATA_PATH'):
        DATA_PATH = os.environ.get("DATA_PATH") + "/"
    else:
        print("Baseline运行时，因为无法读取测试集，所以会有此条报错，属于正常现象")
        print("When baseline is running, this error message will appear because the test set cannot be read, which is a normal phenomenon.")
        #Baseline运行时，因为无法读取测试集，所以会有此条报错，属于正常现象
        #When baseline is running, this error message will appear because the test set cannot be read, which is a normal phenomenon.
    testA_path = train_path   #DATA_PATH + "a_seq_testA.csv"  #读取测试集A, read testing setA
    data_testA = pd.read_csv(testA_path)
    testB_path = train_path #DATA_PATH + "a_seq_testB.csv" #读取测试集B,read teseting setB
    data_testB = pd.read_csv(testB_path)
    #--------------开始进行模型回归--Start Parameter Solving-------------#
    p_train, q_train = pqregression(a_seq)
    #p_train = 1.899389
    #q_train = -0.910914

    #validate(a_seq,p_train,q_train)
    pq_scipy_regression(a_seq)

    a_seq,_ =generateData(2.1234, 4.0678, 1, 2, 3)
    pq_scipy_regression(a_seq)
    p_testA, q_testA = pqregression(a_seq)
    p_testB, q_testB = pqregression(a_seq)

    p = [p_train,p_testA,p_testB]
    q = [q_train,q_testA,q_testB]
    #-----保存参数p，q到CSV文件到submission.csv, to save the parameters p and q to the .csv file-----#
    df_params = pd.DataFrame({'p': p, 'q': q})
    #print(df_params)
    csv_file_path = 'submission.csv'
    df_params.to_csv(csv_file_path, index=False)
if __name__ == '__main__':
     main()
     #print(generateData())