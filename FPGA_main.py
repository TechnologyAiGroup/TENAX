import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from TRL import *
from Datasets.data import *
from PUF.XORAPUF import *


def AttackXPUF_by_TRN(PUFNumber=4, PUFLength=64, filename="", num_sample=40 * 1000,
                      train_ratio=0.8, batch_size=10 * 1000, method="Ett", lr=80e-4, rank=[1,1,1,1,1], n_outputs=(1,),save = True,
                      epochs=1000, dropout = 0.0 ,l2_rate = 0.0001,model_seed=42,load_seed = 42):


    # 划分数据集
    train_size = int(num_sample * train_ratio)
    val_size = num_sample - train_size
    phi_length = PUFLength + 1

    np.random.seed(load_seed)
    tf.random.set_seed(load_seed)
    random.seed(load_seed)
    X_train, Y_train, X_val, Y_val = loadData(filename, num_sample, train_ratio, phi_length)

    # 定义模型结构
    class LRModel(tf.keras.Model):
        def __init__(self, n_xor):
            super(LRModel, self).__init__()
            self.linear = tf.keras.layers.Dense(units=n_xor, activation='sigmoid')

        def call(self, inputs):
            x = self.linear(inputs)
            x = 1 - 2 * x
            x = tf.reduce_prod(x, axis=-1, keepdims=True)
            x = (1 - x) / 2
            return x

    class ETTModel(tf.keras.Model):
        def __init__(self, n_xor,dropout, rank, n_outputs):
            super(ETTModel, self).__init__()
            self.layer = ETTRegressionLayer(n_xor=n_xor,dropout = dropout, rank=rank, n_outputs=n_outputs)
            self.sigmoid = tf.keras.layers.Activation('sigmoid')

        def call(self, x):
            x = self.layer(x)
            return self.sigmoid(x)

    np.random.seed(model_seed)  # NumPy
    tf.random.set_seed(model_seed)  # TensorFlow
    random.seed(model_seed)  # Python 内建 random 库

    # 初始化模型、优化器和损失函数
    if method == "LR":
        model = LRModel(n_xor=PUFNumber)
    elif method == "Ett":
        model = ETTModel(n_xor=PUFNumber, dropout=dropout, rank=rank, n_outputs=n_outputs)


    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # 训练过程
    start_time = time.time()
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    best_val_acc = 0
    best_train_acc = 0
    batch_size = int(batch_size)

    if method == "LR":
        log_filename = f"./Save/{method}/Log/FPGA_{PUFLength}_{PUFNumber}_{train_size / 1000:.0f}k_XORAPUF_{batch_size}_{lr:.4f}_epc{epochs}_l2={l2_rate}_{load_seed}_{model_seed}.txt"
    elif method == "Ett":
        log_filename = f"./Save/{method}/Log/FPGA_{PUFLength}_{PUFNumber}_{train_size / 1000:.0f}k_XORAPUF_{batch_size}_{lr:.4f}_{rank}_epc{epochs}_l2={l2_rate}_dp={dropout}_{load_seed}_{model_seed}.txt"


    with open(log_filename, "w") as f:
        for epoch in range(epochs):
            # 训练阶段
            model.trainable = True
            train_loss = 0.0

            #手动切 batch
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
                        print("L2_loss:",l2_loss.numpy())
                        if save == True:
                            f.write(f'L2_loss:{l2_loss.numpy()}\n')
                    loss += l2_loss * l2_rate

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss += loss.numpy() * batch_x.shape[0]
                pred_labels = (preds > 0.5).numpy().astype(np.float32)
                correct_train += np.sum(pred_labels == batch_y)
                total += batch_y.shape[0]

            train_losses.append(train_loss / train_size)
            if l2_loss.numpy() < 1e-4:
                print("L2_loss less than 1e-4, stop training.")
                if save == True:
                    f.write("L2_loss less than 1e-4, stop training.\n")
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
            if save:
                f.write(
                    f"Epoch {epoch + 1}: Train Loss: {train_loss / train_size:.4f}, Acc: {acc_train:.2%} , Val Loss: {val_loss / val_size:.4f}, Acc: {acc_val:.2%}\n")

        end_time = time.time()
        print(f"Runtime: {(end_time - start_time) / 60:.4f} min")
        print(f"Best Train Acc: {best_train_acc:.2%}")
        print(f"Best Val Acc: {best_val_acc:.2%}")
        if save:
            f.write(f"Runtime: {(end_time - start_time) / 60:.4f} min\n")
            f.write(f"Best Train Acc: {best_train_acc:.2%}\n")
            f.write(f"Best Val Acc: {best_val_acc:.2%}\n")

    if not save and os.path.exists(log_filename):
        os.remove(log_filename)

    name, ext = os.path.splitext(log_filename)
    if not os.path.exists(f"{name}——{best_val_acc:.2%}{ext}") and save:
        os.rename(log_filename, f"{name}——{best_val_acc:.2%}{ext}")


if __name__ == "__main__":


    # 查看 TensorFlow 是否识别到 GPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # 查看 TensorFlow 设备的详细信息
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    # 配置参数
    PUFNumber = 4
    PUFLength = 128
    train_ratio = 0.95

    if PUFLength == 64:
        num_sample = int(80 * 1000 / train_ratio)

    elif PUFLength == 128:
        num_sample = int(60 * 1000 / train_ratio)

    # 训练参数

    n_outputs = (1,)
    epochs = 1000
    save = True
    save_model = False

    method = "LR"

    load_seeds = [64]
    model_seeds = [99]
    l2_rates = [0]
    lrs = [40e-4,60e-4]
    batch_sizes = [60 * 1000]
    rs = [1]*50
    dropouts = [0]

    for load_seed in load_seeds:
        for model_seed in model_seeds:
            for r in rs:
                for lr in lrs:
                    for batch_size in batch_sizes:
                        for l2_rate in l2_rates:
                            for dropout in dropouts:
                                np.random.seed()
                                load_seed = np.random.randint(100)
                                model_seed = np.random.randint(100)

                                # TT_rank = [1, 1, r, r, r, 1, 1]
                                # TT_rank = [1, 1, r, r, 1]
                                TT_rank = [1] + [r] * (PUFNumber - 1) + [1]

                                if method == "Ett":
                                    rank = TT_rank
                                else:
                                    rank = None

                                filename = f"./Datasets/XOR_PUF_128_4_CRPs_transformed.csv"

                                print(
                                    "-----------------------------------Start Attack-----------------------------------")
                                print(
                                    f"PUFNumber: {PUFNumber} ,PUFLength: {PUFLength}  , num_sample:{num_sample}")
                                print(
                                    f"method: {method}, Rank: {rank}, lr: {lr}, batch_size: {batch_size}, epochs: {epochs} , l2_rate : {l2_rate} , dropout : {dropout} ,load_seed :{load_seed}, model_seed :{model_seed}")

                                AttackXPUF_by_TRN(PUFNumber=PUFNumber, PUFLength=PUFLength,
                                                  filename=filename, num_sample=num_sample,
                                                  train_ratio=train_ratio,
                                                  batch_size=batch_size, method=method, lr=lr, rank=rank,
                                                  n_outputs=n_outputs,
                                                  epochs=epochs, dropout=dropout, save=save, l2_rate=l2_rate,
                                                  load_seed=load_seed, model_seed=model_seed)

