import itertools
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import loadmat

from TRL import *
from Datasets.data import *
from PUF.XORAPUF import *
from PUF.LSPUF import *


def AttackLSPUF_by_TRN(PUFNumber=4, PUFLength=64, Noise=0.01, DataSize=80 * 1000, filename="", num_sample=40 * 1000,
                      train_ratio=0.8, batch_size=10 * 1000, method="Ett", lr=80e-4, rank=[1,1,1,1,1], n_outputs=(1,),
                      epochs=1000, dropout = 0.0 ,save=True,l2_rate = 0.0001,model_seed=42,load_seed = 42,data_seed = 42):


    np.random.seed(data_seed)  # NumPy
    tf.random.set_seed(data_seed)  # TensorFlow
    random.seed(data_seed)  # Python 内建 random 库

    # 构建数据集
    LSPUFSample = LSPUF.randomSample(Xnum=PUFNumber, length=PUFLength, alpha=Noise)

    if not os.path.exists(filename):
        makeData_forLSPUF(filename, DataSize, LSPUFSample)

    # 划分数据集
    train_size = int(num_sample * train_ratio)
    val_size = num_sample - train_size
    phi_length = PUFLength + 1

    np.random.seed(load_seed)
    tf.random.set_seed(load_seed)
    random.seed(load_seed)
    X_train, Y_train, X_val, Y_val = loadData_forLSPUF_float32(filename, num_sample, train_ratio)


    class ETTModel(tf.keras.Model):
        def __init__(self, n_xor, dropout, rank, n_outputs):
            super(ETTModel, self).__init__()
            self.layer = ETTRegressionLayer_LSPUF(n_xor=n_xor, dropout=dropout, rank=rank, n_outputs=n_outputs)
            self.sigmoid = tf.keras.layers.Activation('sigmoid')

        def call(self, x):
            x = self.layer(x)
            return self.sigmoid(x)

    np.random.seed(model_seed)  # NumPy
    tf.random.set_seed(model_seed)  # TensorFlow
    random.seed(model_seed)  # Python 内建 random 库

    # 初始化模型、优化器和损失函数
    if method == "Ett":
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


    if method == "Ett":
        log_filename = f"./Save/LSPUF/{method}/Log/{PUFLength}_{PUFNumber}_{train_size / 1000:.0f}k_{Noise}_LSPUF_{batch_size}_{lr:.4f}_{rank}_epc{epochs}_l2={l2_rate}_dp={dropout}_{data_seed}_{load_seed}_{model_seed}.txt"

    with open(log_filename, "w") as f:
        for epoch in range(epochs):
            # 训练阶段
            model.trainable = True
            train_loss = 0.0

            #手动切 batch
            idx = np.arange(train_size)
            np.random.shuffle(idx)

            correct_train = 0
            total = 0
            for start_idx in range(0, train_size, batch_size):
                end_idx = start_idx + batch_size
                batch_x = X_train[idx[start_idx:end_idx]]
                batch_y = Y_train[idx[start_idx:end_idx]]

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

            print(f"Epoch {epoch + 1}: Train Loss: {train_loss / train_size:.4f}, Acc: {acc_train:.2%} , Val Loss: {val_loss / val_size:.4f}, Acc: {acc_val:.2%}")
            if save:
                f.write(f"Epoch {epoch + 1}: Train Loss: {train_loss / train_size:.4f}, Acc: {acc_train:.2%} , Val Loss: {val_loss / val_size:.4f}, Acc: {acc_val:.2%}\n")

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


    if best_val_acc > 0.98 or best_val_acc < 0.7:
        return model



    """ LSPUF : Refine the model found
    please refer to N. Wisiol, G. T. Becker, M. Margraf, T. A. Soroceanu, J. Tobisch, and B. Zengin,
     “Breaking the lightweight secure PUF: Understanding therelation of input transformations and machine learning resistance,”
      in Smart Card Research and Advanced Applications. Cham, Switzerland:Springer, 2020, pp. 40–54.>>
    """

    with open(f"{name}——{best_val_acc:.2%}{ext}","a+") as f:
        start_time = time.time()
        print("---------------------------------------------Start Swapping and Rotating the Cores-----------------------------------------------")
        f.write("---------------------------------------------Start Swapping and Rotating the Cores-----------------------------------------------\n")

        correlation_permutations = loadmat(
            'Datasets/correlation_permutations_lightweight_secure_%i_10.mat' % PUFLength
        )['shiftOverviewData'][:, :, 0].astype('int32')

        model_shift = ETTModel(n_xor=PUFNumber, dropout=dropout, rank=rank, n_outputs=n_outputs)
        model_shift.build(input_shape=(None, X_train.shape[1]))

        # 复制原模型所有参数
        model.trainable = True
        model_shift.trainable = True
        for var_src, var_dst in zip(model.trainable_variables, model_shift.trainable_variables):
            var_dst.assign(var_src)

        # find_high_accuracy_weight_permutations
        high_accuracy_permutations = []
        for permutation in list(itertools.permutations(range(PUFNumber)))[1:]:
            same_shape = True
            for i in range(PUFNumber):
                shifted_core_i = tf.roll(model.layer.core_is[i],-correlation_permutations[i, permutation[i]],axis=2)
                if shifted_core_i.shape != model_shift.layer.core_is[permutation[i]].shape:
                    same_shape = False
                    break
                model_shift.layer.core_is[permutation[i]].assign(shifted_core_i)
            if not same_shape:
                continue
            model_shift.trainable = False
            correct_val = 0
            total = 0
            batch_size_val = min(batch_size, val_size)
            for start_idx in range(0, val_size, batch_size_val):
                end_idx = start_idx + batch_size_val
                val_x = X_val[start_idx:end_idx]
                val_y = Y_val[start_idx:end_idx]

                preds = model_shift(val_x, training=False)
                loss = loss_fn(val_y, preds)

                pred_labels = (preds > 0.5).numpy().astype(np.float32)
                correct_val += np.sum(pred_labels == val_y)
                total += val_y.shape[0]

            acc_val = correct_val / total
            if acc_val > 1.2 * best_val_acc - 0.2:
                high_accuracy_permutations.append((permutation, acc_val))

        high_accuracy_permutations = sorted(high_accuracy_permutations, key=lambda x: x[1], reverse=True)

        # epochs /= 2
        lr *= 2
        # refine the model that found before
        cnt = 1
        final_acc = best_val_acc
        for permutation, acc_val in high_accuracy_permutations[:5 * PUFNumber]:
            print(f"Try No.{cnt} permutation :{permutation} , Accuracy :{acc_val}")
            f.write(f"Try No.{cnt} permutation :{permutation} , Accuracy :{acc_val} \n")
            new_best_val_acc = 0
            best_train_acc = 0


            cnt += 1
            for i in range(PUFNumber):
                shifted_core_i = tf.roll(model.layer.core_is[i],-correlation_permutations[i, permutation[i]],axis=2)
                model_shift.layer.core_is[permutation[i]].assign(shifted_core_i)

            for epoch in range(epochs):
                # 训练阶段
                model_shift.trainable = True
                # 手动切 batch
                idx = np.arange(train_size)
                np.random.shuffle(idx)

                correct_train = 0
                total = 0
                train_loss = 0
                for start_idx in range(0, train_size, batch_size):
                    end_idx = start_idx + batch_size
                    batch_x = X_train[idx[start_idx:end_idx]]
                    batch_y = Y_train[idx[start_idx:end_idx]]

                    with tf.GradientTape() as tape:
                        preds = model_shift(batch_x, training=True)
                        # preds = batch_y
                        loss = loss_fn(batch_y, preds)
                        # 手动加上L2正则项
                        l2_loss = tf.add_n(
                            [tf.nn.l2_loss(var) for var in model_shift.trainable_variables if 'bias' not in var.name])
                        if start_idx == 0:
                            print("L2_loss:", l2_loss.numpy())
                            if save == True:
                                f.write(f'L2_loss:{l2_loss.numpy()}\n')
                        loss += l2_loss * l2_rate

                    gradients = tape.gradient(loss, model_shift.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model_shift.trainable_variables))
                    train_loss += loss.numpy() * batch_x.shape[0]
                    pred_labels = (preds > 0.5).numpy().astype(np.float32)
                    correct_train += np.sum(pred_labels == batch_y)
                    total += batch_y.shape[0]

                acc_train = correct_train / total
                best_train_acc = max(best_train_acc, acc_train)

                # 验证阶段
                model_shift.trainable = False
                val_loss = 0.0
                correct_val = 0
                total = 0
                for start_idx in range(0, val_size, batch_size):
                    end_idx = start_idx + batch_size
                    val_x = X_val[start_idx:end_idx]
                    val_y = Y_val[start_idx:end_idx]

                    preds = model_shift(val_x, training=False)
                    loss = loss_fn(val_y, preds)
                    val_loss += loss.numpy() * val_x.shape[0]

                    pred_labels = (preds > 0.5).numpy().astype(np.float32)
                    correct_val += np.sum(pred_labels == val_y)
                    total += val_y.shape[0]

                acc_val = correct_val / total
                new_best_val_acc = max(acc_val, new_best_val_acc)

                print(f"Epoch {epoch + 1}: Train Loss: {train_loss / train_size:.4f}, Acc: {acc_train:.2%} , Val Loss: {val_loss / val_size:.4f}, Acc: {acc_val:.2%}")
                if save:
                    f.write(f"Epoch {epoch + 1}: Train Loss: {train_loss / train_size:.4f}, Acc: {acc_train:.2%} , Val Loss: {val_loss / val_size:.4f}, Acc: {acc_val:.2%}\n")

            print(f"New Best Train Acc: {best_train_acc:.2%}")
            print(f"New Best Val Acc: {new_best_val_acc:.2%}")
            if save:
                f.write(f"New Best Train Acc: {best_train_acc:.2%}\n")
                f.write(f"New Best Val Acc: {new_best_val_acc:.2%}\n")
            if new_best_val_acc > 0.1 + 0.9 * best_val_acc and new_best_val_acc > final_acc:
                best_model = model_shift
                final_acc = new_best_val_acc

            if final_acc > 0.97:
                break

        end_time = time.time()
        print(f"Refine model RunTime: {(end_time - start_time) / 60:.2f} min")
        print(f"Final Best Val Acc: {final_acc:.2%}")
        if save :
            f.write(f"Refine model RunTime: {(end_time - start_time) / 60:.2f} min\n")
            f.write(f"Final Best Val Acc: {final_acc:.2%}\n")

    if not os.path.exists(f"{name}——{best_val_acc:.2%}——{final_acc:.2%}{ext}") and save:
        os.rename(f"{name}——{best_val_acc:.2%}{ext}", f"{name}——{best_val_acc:.2%}——{final_acc:.2%}{ext}")




if __name__ == "__main__":


    # 查看 TensorFlow 是否识别到 GPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # 查看 TensorFlow 设备的详细信息
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    # 配置参数
    PUFNumber = 4
    PUFLength = 64
    Noise = 0.01
    if PUFLength == 64:
        if PUFNumber == 4:
            DataSize = 200 * 1000
        elif PUFNumber == 5:
            DataSize = 400 * 1000
        elif PUFNumber == 6:
            DataSize = 2000 * 1000

        if PUFNumber == 4:
            num_sample = int(32 * 1000 / 0.95)
        elif PUFNumber == 5:
            num_sample = int(160 * 1000 / 0.95)
        elif PUFNumber == 6:
            num_sample = int(480 * 1000 / 0.95)

    elif PUFLength == 128:
        if PUFNumber == 4:
            DataSize = 1100 * 1000
        elif PUFNumber == 5:
            DataSize = 2200 * 1000

        if PUFNumber == 4:
            num_sample = int(120 * 1000 / 0.95)
        elif PUFNumber == 5:
            num_sample = int(800 * 1000 / 0.95)

    # 训练参数
    train_ratio = 0.95
    n_outputs = (1,)
    epochs = 500
    save = True
    save_model = False

    method = "Ett"

    # DataSize = 1600 * 1000
    # num_sample = int(800 * 1000 / train_ratio)
    # # # 6-xor search
    data_seeds = [42]
    load_seeds = [6]
    model_seeds = [70]
    l2_rates = [0]
    lrs = [60e-4,80e-4]
    batch_sizes = [32 * 1000]
    rs = [1]*50
    dropouts = [0]

for data_seed in data_seeds:
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

                                    TT_rank = [1, 1, r, r, r, 1, 1]
                                    # TT_rank = [1, 1, r, r, 1, 1]
                                    # TT_rank = [1, 1, r, r, 1]
                                    TT_rank = [1] + [r] * (PUFNumber - 1) + [1]

                                    if method == "Ett":
                                        rank = TT_rank
                                    else:
                                        rank = None

                                    filename = f"./Datasets/{PUFLength}_{PUFNumber}_{DataSize / 1000:.0f}k_{Noise}_{data_seed}_LSPUF.csv"
                                    exist = False
                                    for fname in os.listdir(f"./Save/LSPUF/{method}/Log/"):
                                        if fname.startswith(
                                                f"{PUFLength}_{PUFNumber}_{int(num_sample * train_ratio) / 1000:.0f}k_{Noise}_LSPUF_{batch_size}_{lr:.4f}_{rank}_epc{epochs}_l2={l2_rate}_dp={dropout}_{data_seed}_{load_seed}_{model_seed}"):
                                            print("Already Trained")
                                            exist = True
                                            break
                                    if exist:
                                        continue

                                    print("-----------------------------------Start Attack-----------------------------------")
                                    print(f"PUFNumber: {PUFNumber} ,PUFLength: {PUFLength} ,Noise: {Noise} , DataSize:{DataSize} , num_sample:{num_sample}")
                                    print(f"method: {method}, Rank: {rank}, lr: {lr}, batch_size: {batch_size}, epochs: {epochs} , l2_rate : {l2_rate} , dropout : {dropout} , data_seed :{data_seed},load_seed :{load_seed}, model_seed :{model_seed}")

                                    AttackLSPUF_by_TRN(PUFNumber=PUFNumber, PUFLength=PUFLength, Noise=Noise,
                                                          DataSize=DataSize,
                                                          filename=filename, num_sample=num_sample,
                                                          train_ratio=train_ratio,
                                                          batch_size=batch_size, method=method, lr=lr, rank=rank,
                                                          epochs=epochs, dropout=dropout, l2_rate=l2_rate,
                                                          load_seed=load_seed,
                                                          model_seed=model_seed, data_seed=data_seed)

