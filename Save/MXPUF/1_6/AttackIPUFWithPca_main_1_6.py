import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from TRL import *
from Datasets.data import *
from PUF.XORAPUF import *
from PUF.IPUF import *


def SetSeed(seed):
    np.random.seed(seed)  # NumPy
    tf.random.set_seed(seed)  # TensorFlow
    random.seed(seed)  # Python 内建 random 库


def predict(model, batch_C, mu, coeff):
    Phi = np.zeros(shape=(batch_C.shape[0], batch_C.shape[1] + 1))
    for i in range(batch_C.shape[0]):
        Phi[i] = APUF.transform(batch_C[i])

    batch_C = Phi
    batch_C[batch_C == -1] = 0
    batch_C_centered = batch_C - mu
    batch_score = np.dot(batch_C_centered, coeff)
    batch_C_new = np.concatenate([Phi, batch_score], axis=1)
    batch_R = model(batch_C_new, training=False)
    return batch_R


def Interpose(C, insert_col):
    m, n = C.shape
    mid_col = n // 2
    C_d = np.zeros((m, n + 1), dtype=C.dtype)
    C_d[:, :mid_col] = C[:, :mid_col].copy()
    C_d[:, mid_col] = insert_col[:, 0]
    C_d[:, mid_col + 1:] = C[:, mid_col:].copy()
    return C_d


def Heuristic(C, R, model, batch_size, mu, coeff):
    m, n = C.shape
    mid_col = n // 2

    # create c+ and c-
    C_insert_1 = np.zeros((m, n + 1), dtype=C.dtype)
    C_insert_1[:, :mid_col] = C[:, :mid_col].copy()
    C_insert_1[:, mid_col] = 1
    C_insert_1[:, mid_col + 1:] = C[:, mid_col:].copy()

    C_insert_0 = C_insert_1.copy()
    C_insert_0[:, mid_col] = 0

    # create f_d(c+) f_d(c-)
    R_insert_1 = np.zeros((m, 1), dtype=R.dtype)
    R_insert_0 = np.zeros((m, 1), dtype=R.dtype)

    for start_idx in range(0, m, batch_size):
        end_idx = start_idx + batch_size
        batch_C_1 = C_insert_1[start_idx:end_idx].copy()
        batch_C_0 = C_insert_0[start_idx:end_idx].copy()

        batch_R_1 = predict(model, batch_C_1, mu, coeff)
        batch_R_0 = predict(model, batch_C_0, mu, coeff)

        R_insert_1[start_idx:end_idx] = (batch_R_1 > 0.5).numpy().astype(np.float32)
        R_insert_0[start_idx:end_idx] = (batch_R_0 > 0.5).numpy().astype(np.float32)

    # indices of the rows that f_d(c+) not equal to f_d(c-)
    selected_indices = []
    for i in range(m):
        if R_insert_1[i] != R_insert_0[i]:
            selected_indices.append(i)

    # select the rows
    C_H = C[selected_indices, :].copy()
    # create the R_H
    R_H = R[selected_indices, :].copy()
    # set the R_H
    for i in range(len(selected_indices)):
        index = selected_indices[i]
        if R_insert_1[index] == R[index]:
            R_H[i] = 1
        else:
            R_H[i] = 0

    return C_H, R_H


def AttackXPUF_PCA(PUFNumber=4, PUFLength=64, X=None, Y=None,
                   train_ratio=0.8
                   , batch_size=10 * 1000, method="Ett", lr=100e-4, rank=None, n_outputs=(1,),
                   epochs=1000, dropout=0.0, l2_rate=0.0001, pca_components=None, model_seed=42, load_seed=42):
    print(
        f"PUFNumber: {PUFNumber} ,PUFLength: {PUFLength} ,Noise: {Noise} , DataSize:{DataSize} , num_sample:{X.shape[0]}")
    print(
        f"method: {method}, Rank: {rank}, lr: {lr}, batch_size: {batch_size}, epochs: {epochs} , l2_rate : {l2_rate} , dropout : {dropout} , data_seed :{data_seed},load_seed :{load_seed}, model_seed :{model_seed}")

    # 划分数据集
    num_sample = X.shape[0]
    train_size = int(num_sample * train_ratio)
    val_size = num_sample - train_size
    phi_length = PUFLength + 1

    SetSeed(load_seed)
    X_train, Y_train, X_val, Y_val, mu, coeff = loadData_with_PCA_forMXPUF(X, Y, num_sample, train_ratio, phi_length,
                                                                           pca_components=None)

    # 定义模型结构
    class ETTModel(tf.keras.Model):
        def __init__(self, n_xor, dropout, rank, n_outputs):
            super(ETTModel, self).__init__()
            self.layer = ETTRegressionLayer(n_xor=n_xor, dropout=dropout, rank=rank, n_outputs=n_outputs)
            self.sigmoid = tf.keras.layers.Activation('sigmoid')

        def call(self, x):
            x = self.layer(x)
            return self.sigmoid(x)

    SetSeed(model_seed)

    # 初始化模型、优化器和损失函数
    if method == "Ett":
        model = ETTModel(n_xor=PUFNumber, dropout=dropout, rank=rank, n_outputs=n_outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # 训练过程
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    best_val_acc = 0
    best_train_acc = 0
    batch_size = int(batch_size)

    for epoch in range(epochs):
        # 训练阶段
        model.trainable = True
        train_loss = 0.0

        # 手动切 batch
        idx = np.arange(train_size)
        np.random.shuffle(idx)
        X_train = X_train[idx]
        Y_train = Y_train[idx]

        correct_train = 0
        total = 0
        for start_idx in range(0, train_size, batch_size):
            end_idx = start_idx + batch_size
            batch_x = X_train[start_idx:end_idx]
            batch_y = Y_train[start_idx:end_idx]

            with tf.GradientTape() as tape:
                preds = model(batch_x, training=True)
                # preds = batch_y
                loss = loss_fn(batch_y, preds)
                # 手动加上L2正则项
                l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in model.trainable_variables if 'bias' not in var.name])
                if start_idx == 0:
                    # for var in model.trainable_variables:
                    #     print(tf.shape(var))
                    #     print(var.name,":",tf.nn.l2_loss(var).numpy())
                    print("L2_loss:", l2_loss.numpy())

                loss += l2_loss * l2_rate

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss += loss.numpy() * batch_x.shape[0]
            pred_labels = (preds > 0.5).numpy().astype(np.float32)
            correct_train += np.sum(pred_labels == batch_y)
            total += batch_y.shape[0]

        train_losses.append(train_loss / train_size)
        if l2_loss.numpy() < 1e-3:
            print("L2_loss less than 1e-3, stop training.")
            break

        acc_train = correct_train / total
        best_train_acc = max(best_train_acc, acc_train)
        train_acc.append(acc_train)

        # 验证阶段
        model.trainable = False
        val_loss = 0.0
        correct_val = 0
        total = 0
        for start_idx in range(0, val_size, batch_size):
            end_idx = start_idx + batch_size
            val_x = X_val[start_idx:end_idx]
            val_y = Y_val[start_idx:end_idx]

            preds = model(val_x, training=False)
            loss = loss_fn(val_y, preds)
            val_loss += loss.numpy() * val_x.shape[0]

            pred_labels = (preds > 0.5).numpy().astype(np.float32)
            correct_val += np.sum(pred_labels == val_y)
            total += val_y.shape[0]

        val_losses.append(val_loss / val_size)

        acc_val = correct_val / total

        best_val_acc = max(acc_val, best_val_acc)
        val_acc.append(acc_val)

        print(
            f"Epoch {epoch + 1}: Train Loss: {train_loss / train_size:.4f}, Acc: {acc_train:.2%} , Val Loss: {val_loss / val_size:.4f}, Acc: {acc_val:.2%}")

    print(f"Best Train Acc: {best_train_acc:.2%}")
    print(f"Best Val Acc: {best_val_acc:.2%}")

    return model, mu, coeff, best_val_acc


def predict_and_interpose(model_u, C, mu_u, coeff_u, batch_size):
    m = C.shape[0]
    f_u_C = np.zeros((m, 1), dtype=np.float32)
    for start_idx in range(0, m, batch_size):
        end_idx = start_idx + batch_size
        f_u_C_batch = predict(model_u, C[start_idx:end_idx, :].copy(), mu_u, coeff_u)
        f_u_C[start_idx:end_idx] = (f_u_C_batch > 0.5).numpy().astype(np.float32)
    return Interpose(C, f_u_C)


def test_accuracy(model_u, model_d, C_test, R_test, mu_u, coeff_u, mu_d, coeff_d):
    f_u_C_test = predict(model_u, C_test, mu_u, coeff_u)
    C_d_test = Interpose(C_test, (f_u_C_test > 0.5).numpy().astype(np.float32))
    pred_test = predict(model_d, C_d_test, mu_d, coeff_d)
    pred_labels = (pred_test > 0.5).numpy().astype(np.float32)
    return np.sum(pred_labels == R_test) / len(R_test)


def train_and_load_model_u(C, R, model_d, mu_d, coeff_d, batch_size, epochs, Ku, n, cnt, f,
                           lr=1e-2, TT_rank=None, dropout=0, l2_rate=0,
                           load_seed=15, model_seed=72, save_dir="Save/MXPUF/Models", Saved=False):
    if TT_rank is None:
        TT_rank = [1] * 7

    C_u, R_u = Heuristic(C, R, model_d, batch_size, mu_d, coeff_d)

    model_path = f"{save_dir}/Model_1_6_{cnt}_{load_seed}_{model_seed}_up"
    best_val_acc = 0
    if not Saved:
        model_u, mu_u, coeff_u, best_val_acc = AttackXPUF_PCA(
            PUFNumber=Ku, PUFLength=n,
            X=C_u, Y=R_u,
            train_ratio=0.95,
            batch_size=int(min(C_u.shape[0] * 0.95, 800 * 1000)),
            method="Ett", lr=lr,
            rank=TT_rank, n_outputs=(1,),
            epochs=epochs,
            dropout=dropout,
            l2_rate=l2_rate,
            pca_components=None,
            model_seed=model_seed,
            load_seed=load_seed
        )
        f.write(
            f"{cnt}_up 's best acc = {best_val_acc:.2%}, lr={lr:.4f}, TT_rank={TT_rank}, dropout={dropout}, l2={l2_rate}, load_seed={load_seed}, model_seed={model_seed}\n")
        print(
            f"{cnt}_up 's best acc = {best_val_acc:.2%}, lr={lr:.4f}, TT_rank={TT_rank}, dropout={dropout}, l2={l2_rate}, load_seed={load_seed}, model_seed={model_seed}")

        if best_val_acc > 0.65:
            model_u.save(model_path)
            np.savez(f"{model_path}.npz", mu_u=mu_u, coeff_u=coeff_u)
    #     else:
    #            raise RuntimeError("model_u training accuracy too low, exit.")
    else:
        model_u = tf.keras.models.load_model(model_path)

    if best_val_acc > 0.65 or Saved:
        pca_data = np.load(f"{model_path}.npz")
        mu_u = pca_data['mu_u']
        coeff_u = pca_data['coeff_u']
    else:
        mu_u = None
        coeff_u = None

    return model_u, mu_u, coeff_u, best_val_acc


def train_and_load_model_d(C, R, model_u, mu_u, coeff_u, batch_size, epochs, Kd, n, cnt, f,
                           lr=8e-3, TT_rank=None, dropout=0, l2_rate=1e-4,
                           load_seed=36, model_seed=18, save_dir="Save/MXPUF/Models", Saved=False):
    if TT_rank is None:
        TT_rank = [1] * 7

    C_d = predict_and_interpose(model_u, C, mu_u, coeff_u, batch_size)

    model_path = f"{save_dir}/Model_1_6_{cnt}_{load_seed}_{model_seed}_down"
    best_val_acc = 0
    if not Saved:
        model_d, mu_d, coeff_d, best_val_acc = AttackXPUF_PCA(
            PUFNumber=Kd,
            PUFLength=n + 1,
            X=C_d,
            Y=R,
            train_ratio=0.8,
            batch_size=batch_size,
            method="Ett",
            lr=lr,
            rank=TT_rank,
            n_outputs=(1,),
            epochs=epochs,
            dropout=dropout,
            l2_rate=l2_rate,
            pca_components=None,
            model_seed=model_seed,
            load_seed=load_seed
        )
        f.write(
            f"{cnt}_down 's best acc = {best_val_acc:.2%}, lr={lr:.4f}, TT_rank={TT_rank}, dropout={dropout}, l2={l2_rate}, load_seed={load_seed}, model_seed={model_seed}\n")
        print(
            f"{cnt}_down 's best acc = {best_val_acc:.2%}, lr={lr:.4f}, TT_rank={TT_rank}, dropout={dropout}, l2={l2_rate}, load_seed={load_seed}, model_seed={model_seed}")

        if best_val_acc > 0.65:
            model_d.save(model_path)
            np.savez(f"{model_path}.npz", mu_d=mu_d, coeff_d=coeff_d)
    #        else:
    #           raise RuntimeError("model_d training accuracy too low, exit.")
    else:
        model_d = tf.keras.models.load_model(model_path)

    if best_val_acc > 0.65 or Saved:
        pca_data = np.load(f"{model_path}.npz")
        mu_d = pca_data['mu_d']
        coeff_d = pca_data['coeff_d']
    else:
        mu_d = None
        coeff_d = None

    return model_d, mu_d, coeff_d, best_val_acc


def AttackMXPUF_PCA(Ku=1, Kd=4, PUFLength=64, Noise=0.01, DataSize=80 * 1000, filename="", num_sample=40 * 1000,
                    save=True, data_seed=42):
    np.random.seed(data_seed)  # NumPy
    tf.random.set_seed(data_seed)  # TensorFlow
    random.seed(data_seed)  # Python 内建 random 库

    # 构建数据集
    IPUFSample = IPUF.randomSample(num_upper=Ku, num_lower=Kd, length=PUFLength, insert_index=PUFLength // 2,
                                   noise_level=Noise)

    if not os.path.exists(filename):
        makeData_forMXPUF(filename, DataSize, IPUFSample)

    data = np.loadtxt(filename, delimiter=",")
    assert data.shape[0] >= num_sample + 20000, "num_sample Error"
    SetSeed(43)
    np.random.shuffle(data)

    C = data[:num_sample, :PUFLength]
    R = data[:num_sample, -1].reshape(num_sample, 1)
    C_test = data[num_sample:num_sample + 20000, :PUFLength]
    R_test = data[num_sample:num_sample + 20000, -1].reshape(20000, 1)

    m, n = C.shape
    random_bits = np.random.randint(0, 2, size=(m, 1))  # shape (m, 1)
    C_d = Interpose(C, random_bits)

    # 训练参数

    epochs = 500
    start_time = time.time()
    load_seeds = [28]
    model_seeds = [33]
    rs = [1]
    lrs = [100e-4]
    l2_rates = [0]
    dropouts = [0]
    for load_seed in load_seeds:
        for model_seed in model_seeds:
            for r in rs:
                for lr in lrs:
                    for l2_rate in l2_rates:
                        for dropout in dropouts:
                            TT_rank = [1] * 7
                            np.random.seed()
                            load_seed = 24
                            model_seed = 64

                            batch_size = int(min(800 * 1000, 800 * 1000))
                            log_filename = f"./Save/MXPUF/Ett_PCA/Log/{PUFLength}_{Ku}_{Kd}_{int(num_sample * 0.8) / 1000:.0f}k_{Noise}_{batch_size}_{lr:.4f}_{TT_rank}_epc{epochs}_l2={l2_rate}_dp={dropout}_{data_seed}_{load_seed}_{model_seed}_firstmodel.txt"

                            exist = False
                            for fname in os.listdir(f"./Save/MXPUF/Ett_PCA/Log/"):
                                if fname.startswith(
                                        f"{PUFLength}_{Ku}_{Kd}_{int(num_sample * 0.8) / 1000:.0f}k_{Noise}_{batch_size}_{lr:.4f}_{TT_rank}_epc{epochs}_l2={l2_rate}_dp={dropout}_{data_seed}_{load_seed}_{model_seed}_firstmodel"):
                                    print("Already Trained")
                                    exist = True
                                    break
                            # if exist:
                            #     continue

                            saved = False
                            if not saved:
                                model_d, mu_d, coeff_d, best_val_acc = AttackXPUF_PCA(PUFNumber=Kd, PUFLength=n + 1,
                                                                                      X=C_d, Y=R,
                                                                                      train_ratio=0.8,
                                                                                      batch_size=batch_size,
                                                                                      method="Ett",
                                                                                      lr=lr, rank=TT_rank,
                                                                                      n_outputs=(1,),
                                                                                      epochs=epochs,
                                                                                      dropout=dropout,
                                                                                      l2_rate=l2_rate,
                                                                                      pca_components=None,
                                                                                      model_seed=model_seed,
                                                                                      load_seed=load_seed)

                                with open(log_filename, "w") as f:
                                    f.write(f"{best_val_acc:.2%}")
                                name, ext = os.path.splitext(log_filename)
                                if not os.path.exists(f"{name}——{best_val_acc:.2%}{ext}"):
                                    os.rename(log_filename, f"{name}——{best_val_acc:.2%}{ext}")

                                if best_val_acc < 0.63:
                                    continue
                                else:
                                    model_d.save("Save/MXPUF/Models/Model_1_6")
                                    np.savez("Save/MXPUF/Models/mu_coeff_1_6.npz", mu_d=mu_d, coeff_d=coeff_d)

                            acc_test = 0
                            log_filename = f"./Save/MXPUF/Ett_PCA/Log/{PUFLength}_{Ku}_{Kd}_{int(num_sample * 0.8) / 1000:.0f}k_{Noise}_{batch_size}_{lr:.4f}_{TT_rank}_epc{epochs}_l2={l2_rate}_dp={dropout}_{data_seed}_{load_seed}_{model_seed}.txt"

                            with open(log_filename, "w") as f:

                                # 载入初始model_d和PCA参数（只第一次需要）
                                model_d = tf.keras.models.load_model("Save/MXPUF/Models/Model_1_6")
                                pca_data = np.load("Save/MXPUF/Models/mu_coeff_1_6.npz")
                                mu_d = pca_data['mu_d']
                                coeff_d = pca_data['coeff_d']

                                # -------------------------------------------------------------- cnt = 1 --------------------------------------------------------------------
                                cnt = 1
                                Saved = False
                                lr_u = 100e-4
                                TT_rank_u = [1, 1, 1, 1, 1, 1, 1]
                                dropout_u = 0
                                l2_rate_u = 0
                                load_seed_u = 1
                                model_seed_u = 25
                                for num in range(50):
                                    np.random.seed()
                                    load_seed_u = np.random.randint(100)
                                    model_seed_u = np.random.randint(100)

                                    model_u, mu_u, coeff_u, best_val_acc = train_and_load_model_u(
                                        C, R, model_d, mu_d, coeff_d, batch_size, epochs,
                                        Ku, n, cnt, f,
                                        lr=lr_u,
                                        TT_rank=TT_rank_u,
                                        dropout=dropout_u,
                                        l2_rate=l2_rate_u,
                                        load_seed=load_seed_u,
                                        model_seed=model_seed_u,
                                        save_dir="Save/MXPUF/Models",
                                        Saved=Saved
                                    )
                                    if best_val_acc > 0.75:
                                        f.write(
                                            f"{best_val_acc:.2%}, {cnt} up,{lr_u},{TT_rank_u},{dropout_u},{l2_rate_u},{load_seed_u},{model_seed_u} \n")
                                        temp_load_seed_u = load_seed_u
                                        temp_model_seed_u = model_seed_u
                                        break
                                    elif Saved:
                                        break

                                lr_d = 100e-4
                                TT_rank_d = [1, 1, 1, 1, 1, 1, 1]
                                dropout_d = 0
                                l2_rate_d = 0
                                load_seed_d = 24
                                model_seed_d = 64
                                for num in range(50):
                                    if num >= 1 :
                                        np.random.seed()
                                        load_seed_d = np.random.randint(100)
                                        model_seed_d = np.random.randint(100)
                                    model_d, mu_d, coeff_d, best_val_acc = train_and_load_model_d(
                                        C, R, model_u, mu_u, coeff_u, batch_size, epochs,
                                        Kd, n, cnt, f,
                                        lr=lr_d,
                                        TT_rank=TT_rank_d,
                                        dropout=dropout_d,
                                        l2_rate=l2_rate_d,
                                        load_seed=load_seed_d,
                                        model_seed=model_seed_d,
                                        save_dir="Save/MXPUF/Models",
                                        Saved=Saved
                                    )
                                    if best_val_acc > 0.75:
                                        f.write(
                                            f"{best_val_acc:.2%}, {cnt} down,{lr_d},{TT_rank_d},{dropout_d},{l2_rate_d},{load_seed_d},{model_seed_d} \n")
                                        temp_load_seed_d = load_seed_d
                                        temp_model_seed_d = model_seed_d
                                        break
                                    elif Saved:
                                        break

                                # 测试准确率
                                acc_test = test_accuracy(
                                    model_u=model_u, model_d=model_d,
                                    C_test=C_test, R_test=R_test,
                                    mu_u=mu_u, coeff_u=coeff_u,
                                    mu_d=mu_d, coeff_d=coeff_d
                                )

                                print(f"No.{cnt} : Test Accuracy: {acc_test:.2%}")
                                if save:
                                    f.write(f"No.{cnt} : Test Accuracy: {acc_test:.2%}\n")
                                # -------------------------------------------------------------- cnt = 2 --------------------------------------------------------------------
                                cnt = 2
                                Saved = False
                                lr_u = 100e-4
                                TT_rank_u = [1, 1, 1, 1, 1, 1, 1]
                                dropout_u = 0
                                l2_rate_u = 0
                                for num in range(50):
                                    if num == 0:
                                        load_seed_u = temp_load_seed_u
                                        model_seed_u = temp_model_seed_u
                                    else:
                                        np.random.seed()
                                        load_seed_u = np.random.randint(100)
                                        model_seed_u = np.random.randint(100)

                                    model_u, mu_u, coeff_u, best_val_acc = train_and_load_model_u(
                                        C, R, model_d, mu_d, coeff_d, batch_size, epochs,
                                        Ku, n, cnt, f,
                                        lr=lr_u,
                                        TT_rank=TT_rank_u,
                                        dropout=dropout_u,
                                        l2_rate=l2_rate_u,
                                        load_seed=load_seed_u,
                                        model_seed=model_seed_u,
                                        save_dir="Save/MXPUF/Models",
                                        Saved=Saved
                                    )
                                    if best_val_acc > 0.75:
                                        f.write(
                                            f"{best_val_acc:.2%}, {cnt} up,{lr_u},{TT_rank_u},{dropout_u},{l2_rate_u},{load_seed_u},{model_seed_u} \n")
                                        temp_load_seed_u = load_seed_u
                                        temp_model_seed_u = model_seed_u
                                        break
                                    elif Saved:
                                        break

                                lr_d = 100e-4
                                TT_rank_d = [1, 1, 1, 1, 1, 1, 1]
                                dropout_d = 0
                                l2_rate_d = 0
                                for num in range(50):
                                    if num == 0:
                                        load_seed_d = temp_load_seed_d
                                        model_seed_d = temp_model_seed_d
                                    else:
                                        np.random.seed()
                                        load_seed_d = np.random.randint(100)
                                        model_seed_d = np.random.randint(100)

                                    model_d, mu_d, coeff_d, best_val_acc = train_and_load_model_d(
                                        C, R, model_u, mu_u, coeff_u, batch_size, epochs,
                                        Kd, n, cnt, f,
                                        lr=lr_d,
                                        TT_rank=TT_rank_d,
                                        dropout=dropout_d,
                                        l2_rate=l2_rate_d,
                                        load_seed=load_seed_d,
                                        model_seed=model_seed_d,
                                        save_dir="Save/MXPUF/Models",
                                        Saved=Saved
                                    )
                                    if best_val_acc > 0.75:
                                        f.write(
                                            f"{best_val_acc:.2%}, {cnt} down,{lr_d},{TT_rank_d},{dropout_d},{l2_rate_d},{load_seed_d},{model_seed_d} \n")
                                        temp_load_seed_d = load_seed_d
                                        temp_model_seed_d = model_seed_d
                                        break
                                    elif Saved:
                                        break

                                # 测试准确率
                                acc_test = test_accuracy(
                                    model_u=model_u, model_d=model_d,
                                    C_test=C_test, R_test=R_test,
                                    mu_u=mu_u, coeff_u=coeff_u,
                                    mu_d=mu_d, coeff_d=coeff_d
                                )

                                print(f"No.{cnt} : Test Accuracy: {acc_test:.2%}")
                                if save:
                                    f.write(f"No.{cnt} : Test Accuracy: {acc_test:.2%}\n")
                                # -------------------------------------------------------------- cnt = 3 --------------------------------------------------------------------
                                cnt = 3
                                Saved = False
                                lr_u = 100e-4
                                TT_rank_u = [1, 1, 1, 1, 1, 1, 1]
                                dropout_u = 0
                                l2_rate_u = 0
                                for num in range(50):
                                    if num == 0:
                                        load_seed_u = temp_load_seed_u
                                        model_seed_u = temp_model_seed_u
                                    else:
                                        np.random.seed()
                                        load_seed_u = np.random.randint(100)
                                        model_seed_u = np.random.randint(100)

                                    model_u, mu_u, coeff_u, best_val_acc = train_and_load_model_u(
                                        C, R, model_d, mu_d, coeff_d, batch_size, epochs,
                                        Ku, n, cnt, f,
                                        lr=lr_u,
                                        TT_rank=TT_rank_u,
                                        dropout=dropout_u,
                                        l2_rate=l2_rate_u,
                                        load_seed=load_seed_u,
                                        model_seed=model_seed_u,
                                        save_dir="Save/MXPUF/Models",
                                        Saved=Saved
                                    )
                                    if best_val_acc > 0.75:
                                        f.write(
                                            f"{best_val_acc:.2%}, {cnt} up,{lr_u},{TT_rank_u},{dropout_u},{l2_rate_u},{load_seed_u},{model_seed_u} \n")
                                        temp_load_seed_u = load_seed_u
                                        temp_model_seed_u = model_seed_u
                                        break
                                    elif Saved:
                                        break

                                lr_d = 100e-4
                                TT_rank_d = [1, 1, 1, 1, 1, 1, 1]
                                dropout_d = 0
                                l2_rate_d = 0
                                for num in range(50):
                                    if num == 0:
                                        load_seed_d = temp_load_seed_d
                                        model_seed_d = temp_model_seed_d
                                    else:
                                        np.random.seed()
                                        load_seed_d = np.random.randint(100)
                                        model_seed_d = np.random.randint(100)

                                    model_d, mu_d, coeff_d, best_val_acc = train_and_load_model_d(
                                        C, R, model_u, mu_u, coeff_u, batch_size, epochs,
                                        Kd, n, cnt, f,
                                        lr=lr_d,
                                        TT_rank=TT_rank_d,
                                        dropout=dropout_d,
                                        l2_rate=l2_rate_d,
                                        load_seed=load_seed_d,
                                        model_seed=model_seed_d,
                                        save_dir="Save/MXPUF/Models",
                                        Saved=Saved
                                    )
                                    if best_val_acc > 0.75:
                                        f.write(
                                            f"{best_val_acc:.2%}, {cnt} down,{lr_d},{TT_rank_d},{dropout_d},{l2_rate_d},{load_seed_d},{model_seed_d} \n")
                                        temp_load_seed_d = load_seed_d
                                        temp_model_seed_d = model_seed_d
                                        break
                                    elif Saved:
                                        break

                                # 测试准确率
                                acc_test = test_accuracy(
                                    model_u=model_u, model_d=model_d,
                                    C_test=C_test, R_test=R_test,
                                    mu_u=mu_u, coeff_u=coeff_u,
                                    mu_d=mu_d, coeff_d=coeff_d
                                )

                                print(f"No.{cnt} : Test Accuracy: {acc_test:.2%}")
                                if save:
                                    f.write(f"No.{cnt} : Test Accuracy: {acc_test:.2%}\n")
                                # -------------------------------------------------------------- cnt = 4 --------------------------------------------------------------------
                                cnt = 4
                                Saved = False
                                lr_u = 100e-4
                                TT_rank_u = [1, 1, 1, 1, 1, 1, 1]
                                dropout_u = 0
                                l2_rate_u = 0
                                for num in range(50):
                                    if num == 0:
                                        load_seed_u = temp_load_seed_u
                                        model_seed_u = temp_model_seed_u
                                    else:
                                        np.random.seed()
                                        load_seed_u = np.random.randint(100)
                                        model_seed_u = np.random.randint(100)

                                    model_u, mu_u, coeff_u, best_val_acc = train_and_load_model_u(
                                        C, R, model_d, mu_d, coeff_d, batch_size, epochs,
                                        Ku, n, cnt, f,
                                        lr=lr_u,
                                        TT_rank=TT_rank_u,
                                        dropout=dropout_u,
                                        l2_rate=l2_rate_u,
                                        load_seed=load_seed_u,
                                        model_seed=model_seed_u,
                                        save_dir="Save/MXPUF/Models",
                                        Saved=Saved
                                    )
                                    if best_val_acc > 0.75:
                                        f.write(
                                            f"{best_val_acc:.2%}, {cnt} up,{lr_u},{TT_rank_u},{dropout_u},{l2_rate_u},{load_seed_u},{model_seed_u} \n")
                                        temp_load_seed_u = load_seed_u
                                        temp_model_seed_u = model_seed_u
                                        break
                                    elif Saved:
                                        break

                                lr_d = 100e-4
                                TT_rank_d = [1, 1, 1, 1, 1, 1, 1]
                                dropout_d = 0
                                l2_rate_d = 0
                                for num in range(50):
                                    if num == 0:
                                        load_seed_d = temp_load_seed_d
                                        model_seed_d = temp_model_seed_d
                                    else:
                                        np.random.seed()
                                        load_seed_d = np.random.randint(100)
                                        model_seed_d = np.random.randint(100)

                                    model_d, mu_d, coeff_d, best_val_acc = train_and_load_model_d(
                                        C, R, model_u, mu_u, coeff_u, batch_size, epochs,
                                        Kd, n, cnt, f,
                                        lr=lr_d,
                                        TT_rank=TT_rank_d,
                                        dropout=dropout_d,
                                        l2_rate=l2_rate_d,
                                        load_seed=load_seed_d,
                                        model_seed=model_seed_d,
                                        save_dir="Save/MXPUF/Models",
                                        Saved=Saved
                                    )
                                    if best_val_acc > 0.75:
                                        f.write(
                                            f"{best_val_acc:.2%}, {cnt} down,{lr_d},{TT_rank_d},{dropout_d},{l2_rate_d},{load_seed_d},{model_seed_d} \n")
                                        temp_load_seed_d = load_seed_d
                                        temp_model_seed_d = model_seed_d
                                        break
                                    elif Saved:
                                        break

                                # 测试准确率
                                acc_test = test_accuracy(
                                    model_u=model_u, model_d=model_d,
                                    C_test=C_test, R_test=R_test,
                                    mu_u=mu_u, coeff_u=coeff_u,
                                    mu_d=mu_d, coeff_d=coeff_d
                                )

                                print(f"No.{cnt} : Test Accuracy: {acc_test:.2%}")
                                if save:
                                    f.write(f"No.{cnt} : Test Accuracy: {acc_test:.2%}\n")
                                # -------------------------------------------------------------- cnt = 5 --------------------------------------------------------------------
                                cnt = 5
                                Saved = False
                                lr_u = 100e-4
                                TT_rank_u = [1, 1, 1, 1, 1, 1, 1]
                                dropout_u = 0
                                l2_rate_u = 0
                                for num in range(50):
                                    if num == 0:
                                        load_seed_u = temp_load_seed_u
                                        model_seed_u = temp_model_seed_u
                                    else:
                                        np.random.seed()
                                        load_seed_u = np.random.randint(100)
                                        model_seed_u = np.random.randint(100)

                                    model_u, mu_u, coeff_u, best_val_acc = train_and_load_model_u(
                                        C, R, model_d, mu_d, coeff_d, batch_size, epochs,
                                        Ku, n, cnt, f,
                                        lr=lr_u,
                                        TT_rank=TT_rank_u,
                                        dropout=dropout_u,
                                        l2_rate=l2_rate_u,
                                        load_seed=load_seed_u,
                                        model_seed=model_seed_u,
                                        save_dir="Save/MXPUF/Models",
                                        Saved=Saved
                                    )
                                    if best_val_acc > 0.75:
                                        f.write(
                                            f"{best_val_acc:.2%}, {cnt} up,{lr_u},{TT_rank_u},{dropout_u},{l2_rate_u},{load_seed_u},{model_seed_u} \n")
                                        temp_load_seed_u = load_seed_u
                                        temp_model_seed_u = model_seed_u
                                        break
                                    elif Saved:
                                        break

                                lr_d = 100e-4
                                TT_rank_d = [1, 1, 1, 1, 1, 1, 1]
                                dropout_d = 0
                                l2_rate_d = 0
                                for num in range(50):
                                    if num == 0:
                                        load_seed_d = temp_load_seed_d
                                        model_seed_d = temp_model_seed_d
                                    else:
                                        np.random.seed()
                                        load_seed_d = np.random.randint(100)
                                        model_seed_d = np.random.randint(100)

                                    model_d, mu_d, coeff_d, best_val_acc = train_and_load_model_d(
                                        C, R, model_u, mu_u, coeff_u, batch_size, epochs,
                                        Kd, n, cnt, f,
                                        lr=lr_d,
                                        TT_rank=TT_rank_d,
                                        dropout=dropout_d,
                                        l2_rate=l2_rate_d,
                                        load_seed=load_seed_d,
                                        model_seed=model_seed_d,
                                        save_dir="Save/MXPUF/Models",
                                        Saved=Saved
                                    )
                                    if best_val_acc > 0.75:
                                        f.write(
                                            f"{best_val_acc:.2%}, {cnt} down,{lr_d},{TT_rank_d},{dropout_d},{l2_rate_d},{load_seed_d},{model_seed_d} \n")
                                        temp_load_seed_d = load_seed_d
                                        temp_model_seed_d = model_seed_d
                                        break
                                    elif Saved:
                                        break

                                # 测试准确率
                                acc_test = test_accuracy(
                                    model_u=model_u, model_d=model_d,
                                    C_test=C_test, R_test=R_test,
                                    mu_u=mu_u, coeff_u=coeff_u,
                                    mu_d=mu_d, coeff_d=coeff_d
                                )

                                print(f"No.{cnt} : Test Accuracy: {acc_test:.2%}")
                                if save:
                                    f.write(f"No.{cnt} : Test Accuracy: {acc_test:.2%}\n")


                                end_time = time.time()
                                print(f"Runtime: {(end_time - start_time) / 60:.4f} min")
                                if save:
                                    f.write(f"Runtime: {(end_time - start_time) / 60:.4f} min\n")

                                return model_d, mu_d, coeff_d





if __name__ == "__main__":
    # 查看 TensorFlow 是否识别到 GPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # 查看 TensorFlow 设备的详细信息
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())

    PUFLength = 64
    Ku = 1
    Kd = 6
    DataSize = 4000 * 1000
    Noise = 0
    data_seed = 42
    filename = f"./Datasets/{PUFLength}_{Ku}_{Kd}_{DataSize / 1000:.0f}k_{Noise}_{data_seed}_MXPUF.csv"
    num_sample = int(800 * 1000 / 0.8)
    save = True

    AttackMXPUF_PCA(Ku=Ku, Kd=Kd, PUFLength=PUFLength, Noise=Noise, DataSize=DataSize, filename=filename,
                    num_sample=num_sample,
                    save=save, data_seed=data_seed)