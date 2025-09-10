import tensorflow as tf
import numpy as np
import random
import os
from PUF.APUF import *
import csv



def makeData(filename, dataSize, PUFSample, buffer_size=4000000):
    length = PUFSample.length
    buffer = []

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(dataSize):
            C = np.asarray([random.randint(0, 1) for _ in range(length)])
            phi = PUFSample.transform(C)
            R = PUFSample.getResponse(phi=phi, noisy=True)
            dataline = np.hstack((phi, R)).astype(int)
            buffer.append(dataline)

            if len(buffer) >= buffer_size:
                writer.writerows(buffer)
                buffer = []  # 清空缓冲区

        if buffer:  # 写入剩余数据
            writer.writerows(buffer)



def loadData(filename, num_sample, train_ratio, phi_length):
    data = np.loadtxt(filename, delimiter=",")
    assert data.shape[0] >= num_sample, "num_sample Error"
    np.random.shuffle(data)
    data = data[:num_sample]

    X = data[:, :phi_length]    # (N, phi_length)
    Y = data[:, -1:]            # (N, 1)

    train_size = int(num_sample * train_ratio)
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]

    return X_train, Y_train, X_val, Y_val

def loadData_float32(filename, num_sample, train_ratio, phi_length):
    data = np.loadtxt(filename, delimiter=",",dtype=np.float32)         #this can save a half of memory
    # data = np.loadtxt(filename, delimiter=",")
    assert data.shape[0] >= num_sample, "num_sample Error"
    np.random.shuffle(data)
    data = data[:num_sample]

    X = data[:, :phi_length]    # (N, phi_length)
    Y = data[:, -1:]            # (N, 1)

    train_size = int(num_sample * train_ratio)
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]

    return X_train, Y_train, X_val, Y_val

def loadData_with_PCA(filename, num_sample, train_ratio, phi_length, pca_components=None):
    # 1. 读入数据
    data = np.loadtxt(filename, delimiter=",")
    assert data.shape[0] >= num_sample, "num_sample Error"
    np.random.shuffle(data)
    data = data[:num_sample]


    features = data[:, :phi_length]
    labels = data[:, -1:]
    features_raw = features
    # 2. 特征处理，把-1变成0
    features[features == -1] = 0

    # 3. 手动划分（保持numpy数组）
    train_size = int(num_sample * train_ratio)
    X_train_np = features[:train_size]
    X_train_np_raw = features_raw[:train_size]  #use this to reproduce the experiment in the paper
    # X_train_np_raw = features_raw[:train_size].copy()        #this is more recommended to use
    Y_train_np = labels[:train_size]
    X_val_np = features[train_size:]
    X_val_np_raw = features_raw[train_size:]
    Y_val_np = labels[train_size:]

    coeff_train, score_train, latent_train, tsquared_train, explained_train, mu_train = pca(X_train_np, n_components=pca_components)

    # 5. 拼接原特征 + PCA特征 (训练集)
    X_train_new = np.concatenate([X_train_np_raw, score_train], axis=1)

    # 6. 验证集：用训练集的均值和PCA变换进行转换
    mu_train = np.mean(X_train_np, axis=0)
    X_val_centered = X_val_np - mu_train
    score_val = np.dot(X_val_centered, coeff_train[:, :score_train.shape[1]])  # 使用训练集的主成分
    X_val_new = np.concatenate([X_val_np_raw, score_val], axis=1)

    return X_train_new, Y_train_np, X_val_new, Y_val_np ,mu_train ,coeff_train[:, :score_train.shape[1]]


def loadData_with_PCA_float32(filename, num_sample, train_ratio, phi_length, pca_components=None):     #this can save a half of memory
    # 1. 读入数据
    data = np.loadtxt(filename, delimiter=",",dtype=np.float32)

    assert data.shape[0] >= num_sample, "num_sample Error"
    np.random.shuffle(data)
    data = data[:num_sample]


    features = data[:, :phi_length]
    labels = data[:, -1:]
    features_raw = features
    # 2. 特征处理，把-1变成0
    features[features == -1] = 0

    # 3. 手动划分（保持numpy数组）
    train_size = int(num_sample * train_ratio)
    X_train_np = features[:train_size]
    X_train_np_raw = features_raw[:train_size]
    Y_train_np = labels[:train_size]
    X_val_np = features[train_size:]
    X_val_np_raw = features_raw[train_size:]
    Y_val_np = labels[train_size:]

    coeff_train, score_train, latent_train, tsquared_train, explained_train, mu_train = pca_float32(X_train_np, n_components=pca_components)

    # 5. 拼接原特征 + PCA特征 (训练集)
    X_train_new = np.concatenate([X_train_np_raw, score_train], axis=1,dtype=np.float32)

    # 6. 验证集：用训练集的均值和PCA变换进行转换
    mu_train = np.mean(X_train_np, axis=0)
    X_val_centered = X_val_np - mu_train
    score_val = np.dot(X_val_centered, coeff_train[:, :score_train.shape[1]])  # 使用训练集的主成分
    X_val_new = np.concatenate([X_val_np_raw, score_val], axis=1)

    return X_train_new, Y_train_np, X_val_new, Y_val_np ,mu_train ,coeff_train[:, :score_train.shape[1]]


def loadData_with_PCA_III(filename, num_sample, train_ratio, phi_length, random_select_ratio=0.9, pca_components=None):
    # 1. 读入数据
    data = np.loadtxt(filename, delimiter=",")
    assert data.shape[0] >= num_sample, "num_sample Error"
    np.random.shuffle(data)
    data = data[:num_sample]


    features = data[:, :phi_length]
    labels = data[:, -1:]
    features_raw = features
    # 2. 特征处理，把-1变成0
    features[features == -1] = 0

    # 3. 手动划分（保持numpy数组）
    train_size = int(num_sample * train_ratio)
    X_train_np = features[:train_size]
    X_train_np_raw = features_raw[:train_size]
    Y_train_np = labels[:train_size]
    X_val_np = features[train_size:]
    X_val_np_raw = features_raw[train_size:]
    Y_val_np = labels[train_size:]

    coeff_train, score_train, latent_train, tsquared_train, explained_train, mu_train = pca(X_train_np, n_components=pca_components)

    # 5. 拼接原特征 + PCA特征 (训练集)
    X_train_new = np.concatenate([X_train_np_raw, score_train], axis=1)

    # 6. 验证集：用训练集的均值和PCA变换进行转换
    mu_train = np.mean(X_train_np, axis=0)
    X_val_centered = X_val_np - mu_train
    score_val = np.dot(X_val_centered, coeff_train[:, :score_train.shape[1]])  # 使用训练集的主成分
    X_val_new = np.concatenate([X_val_np_raw, score_val], axis=1)


    # 7. 随机选取
    random_select_length = int(np.shape(X_train_new)[1] * random_select_ratio)

    # 随机选择random_select_length列的索引（不放回），然后排序保持原顺序
    selected_cols = np.sort(np.random.choice(np.shape(X_train_new)[1], size=random_select_length, replace=False))

    return X_train_new[:, selected_cols], Y_train_np, X_val_new[:, selected_cols], Y_val_np,mu_train ,coeff_train[:, :score_train.shape[1]],selected_cols


def makeData_forMXPUF(filename, dataSize, PUFSample):
    dataSet = []
    length = PUFSample.length
    for _ in range(dataSize):
        C = np.asarray([random.randint(0, 1) for _ in range(length)])
        R = PUFSample.getResponse(challenge=C,noisy=True)
        dataline = np.hstack((C, R)).tolist()
        dataSet.append(dataline)
    dataSet = np.asarray(dataSet)
    np.savetxt(filename, dataSet, fmt='%d', delimiter=',')


def loadData_with_forMXPUF(X,Y,num_sample, train_ratio, phi_length):
    Phi = np.zeros(shape=(X.shape[0], X.shape[1]+1))
    for i in range(X.shape[0]):
        Phi[i] = APUF.transform(X[i])
    # 1. 读入数据
    data = np.hstack((Phi, Y))
    assert data.shape[0] >= num_sample, "num_sample Error"
    np.random.shuffle(data)
    data = data[:num_sample]

    X = data[:, :phi_length]  # (N, phi_length)
    Y = data[:, -1:]  # (N, 1)

    train_size = int(num_sample * train_ratio)
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]

    return X_train, Y_train, X_val, Y_val



def loadData_with_PCA_forMXPUF(X,Y,num_sample, train_ratio, phi_length, pca_components=None):
    Phi = np.zeros(shape=(X.shape[0], X.shape[1]+1))
    for i in range(X.shape[0]):
        Phi[i] = APUF.transform(X[i])
    # 1. 读入数据
    data = np.hstack((Phi, Y))
    assert data.shape[0] >= num_sample, "num_sample Error"
    np.random.shuffle(data)
    data = data[:num_sample]


    features = data[:, :phi_length]
    labels = data[:, -1:]
    features_raw = features
    # 2. 特征处理，把-1变成0
    features[features == -1] = 0

    # 3. 手动划分（保持numpy数组）
    train_size = int(num_sample * train_ratio)
    X_train_np = features[:train_size]
    X_train_np_raw = features_raw[:train_size]
    Y_train_np = labels[:train_size]
    X_val_np = features[train_size:]
    X_val_np_raw = features_raw[train_size:]
    Y_val_np = labels[train_size:]

    coeff_train, score_train, latent_train, tsquared_train, explained_train, mu_train = pca(X_train_np, n_components=pca_components)

    # 5. 拼接原特征 + PCA特征 (训练集)
    X_train_new = np.concatenate([X_train_np_raw, score_train], axis=1)

    # 6. 验证集：用训练集的均值和PCA变换进行转换
    mu_train = np.mean(X_train_np, axis=0)
    X_val_centered = X_val_np - mu_train
    score_val = np.dot(X_val_centered, coeff_train[:, :score_train.shape[1]])  # 使用训练集的主成分
    X_val_new = np.concatenate([X_val_np_raw, score_val], axis=1)

    return X_train_new, Y_train_np, X_val_new, Y_val_np ,mu_train ,coeff_train[:, :score_train.shape[1]]



def makeData_forLSPUF(filename, dataSize, PUFSample, buffer_size=1000000):
    length = PUFSample.length
    buffer = []

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(dataSize):
            C = np.asarray([random.randint(0, 1) for _ in range(length)])
            phi = PUFSample.inputNetwork(C)
            R = PUFSample.getResponse(phi = phi, noisy=True)
            dataline = np.hstack((phi.flatten(), R)).astype(int)
            buffer.append(dataline)

            if len(buffer) >= buffer_size:
                writer.writerows(buffer)
                buffer = []  # 清空缓冲区

        if buffer:  # 写入剩余数据
            writer.writerows(buffer)


def loadData_forLSPUF_float32(filename, num_sample, train_ratio):
    data = np.loadtxt(filename, delimiter=",",dtype=np.float32)
    assert data.shape[0] >= num_sample, "num_sample Error"
    np.random.shuffle(data)
    data = data[:num_sample]

    X = data[:, :-1]    # (num_sample, phi_length * n_xor)
    Y = data[:, -1:]            # (num_sample, 1)

    train_size = int(num_sample * train_ratio)
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]


    return X_train, Y_train, X_val, Y_val

def loadData_with_PCA_forLSPUF_float32(filename, num_sample, train_ratio, PUFNumber,phi_length):
    data = np.loadtxt(filename, delimiter=",",dtype=np.float32)
    assert data.shape[0] >= num_sample, "num_sample Error"
    np.random.shuffle(data)
    data = data[:num_sample]

    X = data[:, :-1]    # (num_sample, phi_length * n_xor)
    Y = data[:, -1:]            # (num_sample, 1)

    train_size = int(num_sample * train_ratio)
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]

    X_train_new = np.zeros(shape=(X_train.shape[0],PUFNumber*phi_length*2), dtype=np.float32)
    X_val_new = np.zeros(shape=(X_val.shape[0],PUFNumber*phi_length*2), dtype=np.float32)

    X_train_raw = X_train.copy()
    X_val_raw = X_val.copy()
    X_train[X_train == -1] = 0
    X_val[X_val == -1] = 0

    for cnt in range(PUFNumber):
        phi_train_raw = X_train_raw[:,cnt*phi_length:(cnt+1)*phi_length].copy()
        phi_val_raw = X_val_raw[:,cnt*phi_length:(cnt+1)*phi_length].copy()

        coeff_train,score_train,_,_,_,mu_train = pca_float32(X_train[:,cnt*phi_length:(cnt+1)*phi_length])
        X_train_new[:,cnt*phi_length*2:(cnt+1)*phi_length*2] = np.concatenate([phi_train_raw, score_train], axis=1)

        mu_train = np.mean(X_train[:,cnt*phi_length:(cnt+1)*phi_length],axis=0)
        score_val = np.dot(X_val[:,cnt*phi_length:(cnt+1)*phi_length] - mu_train,coeff_train[:,:score_train.shape[1]])
        X_val_new[:,cnt*phi_length*2:(cnt+1)*phi_length*2] = np.concatenate([phi_val_raw, score_val], axis=1)


    return X_train_new, Y_train, X_val_new, Y_val



def pca(X, n_components=None):
    """
    与MATLAB pca接口一致，返回：coeff, score, latent, tsquared, explained, mu
    输入：
        X: (N, D) 数据矩阵
        n_components: 要保留的主成分个数（None表示保留全部）
    返回：
        coeff: (D, D) or (D, n_components)，主成分方向
        score: (N, D) or (N, n_components)，主成分投影
        latent: (D,) 特征值
        tsquared: (N,) 每个样本的T²统计量
        explained: (D,) 每个主成分解释的方差百分比
        mu: (D,) 每一列特征的均值
    """

    # N: 样本数，D: 特征数
    N, D = X.shape

    # 计算均值并中心化
    mu = np.mean(X, axis=0)
    X_centered = X - mu

    # 协方差矩阵
    cov = np.cov(X_centered, rowvar=False)

    # 特征值分解（eigh适合对称矩阵）
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 排序特征值和特征向量，按降序排列
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    if n_components is not None:
        eigvecs = eigvecs[:, :n_components]
        eigvals = eigvals[:n_components]

    # 得到score
    score = np.dot(X_centered, eigvecs)

    # 计算 T²统计量
    tsquared = np.sum(score ** 2 / eigvals, axis=1)

    # 解释方差百分比
    explained = eigvals / np.sum(eigvals) * 100.0

    return eigvecs, score, eigvals, tsquared, explained, mu



def pca_float32(X, n_components=None):
    """
    与MATLAB pca接口一致，返回：coeff, score, latent, tsquared, explained, mu
    输入：
        X: (N, D) 数据矩阵
        n_components: 要保留的主成分个数（None表示保留全部）
    返回：
        coeff: (D, D) or (D, n_components)，主成分方向
        score: (N, D) or (N, n_components)，主成分投影
        latent: (D,) 特征值
        tsquared: (N,) 每个样本的T²统计量
        explained: (D,) 每个主成分解释的方差百分比
        mu: (D,) 每一列特征的均值
    """

    # N: 样本数，D: 特征数
    N, D = X.shape

    # 计算均值并中心化
    mu = np.mean(X, axis=0,dtype=np.float32)
    X_centered = X - mu

    # 协方差矩阵
    cov = np.cov(X_centered, rowvar=False,dtype=np.float32)

    # 特征值分解（eigh适合对称矩阵）
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 排序特征值和特征向量，按降序排列
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    if n_components is not None:
        eigvecs = eigvecs[:, :n_components]
        eigvals = eigvals[:n_components]

    # 得到score
    score = np.dot(X_centered, eigvecs)

    # 计算 T²统计量
    tsquared = np.sum(score ** 2 / eigvals, axis=1,dtype=np.float32)

    # 解释方差百分比
    explained = eigvals / np.sum(eigvals) * 100.0

    return eigvecs, score, eigvals, tsquared, explained, mu




def processFpgaData_forLSPUF(filename):
    length = PUFSample.length
    buffer = []

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(dataSize):
            C = np.asarray([random.randint(0, 1) for _ in range(length)])
            phi = PUFSample.inputNetwork(C)
            R = PUFSample.getResponse(phi = phi, noisy=True)
            dataline = np.hstack((phi.flatten(), R)).astype(int)
            buffer.append(dataline)

            if len(buffer) >= buffer_size:
                writer.writerows(buffer)
                buffer = []  # 清空缓冲区

        if buffer:  # 写入剩余数据
            writer.writerows(buffer)