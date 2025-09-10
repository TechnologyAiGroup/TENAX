import itertools
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA

from TRL import *
from Datasets.data import *
from PUF.XORAPUF import *
from PUF.LSPUF import *


def AttackLSPUF(PUFNumber=4, PUFLength=64,  filename="", num_sample=40 * 1000,
                      train_ratio=0.8, batch_size=10 * 1000, method="LR", lr=80e-4,
                      epochs=1000,save=True,model_seed=42,load_seed = 42):

    # 划分数据集
    train_size = int(num_sample * train_ratio)
    val_size = num_sample - train_size
    phi_length = PUFLength + 1

    np.random.seed(load_seed)
    tf.random.set_seed(load_seed)
    random.seed(load_seed)
    X_train, Y_train, X_val, Y_val = loadData_forLSPUF_float32(filename, num_sample, train_ratio)

    # 定义模型结构
    class LSPUFLRModel(tf.keras.Model):
        def __init__(self, phi_length, n_xor):
            super(LSPUFLRModel, self).__init__()
            self.phi_length = phi_length
            self.n_xor = n_xor

            # 每个 block 对应一个独立的 weight 向量
            # shape: (n_xor, phi_length)
            initializer = tf.keras.initializers.GlorotUniform()
            self.puf_weights = tf.Variable(
                initializer(shape=(n_xor, phi_length)),
                trainable=True, name="LSPUF_weights"
            )

        def call(self, inputs):  # inputs shape: (batch_size, phi_length * n_xor)
            batch_size = tf.shape(inputs)[0]

            # reshape 为 (batch_size, n_xor, phi_length)
            x = tf.reshape(inputs, (batch_size, self.n_xor, self.phi_length))

            # 每个 block 与对应 weight 做逐块内积，广播计算
            # → result shape: (batch_size, n_xor)
            logits = tf.reduce_sum(x * self.puf_weights[None, :, :], axis=-1)

            # sigmoid → ∈ (0, 1)，再转为 {-1, +1}
            x = 1 - 2 * tf.sigmoid(logits)

            # 按 XOR 逻辑组合
            # XOR = prod(-1/+1) → ∈ {-1, +1} → map to {0, 1}
            x = tf.reduce_prod(x, axis=-1, keepdims=True)  # shape: (batch_size, 1)
            x = (1 - x) / 2  # → ∈ {0, 1}
            return x


    np.random.seed(model_seed)  # NumPy
    tf.random.set_seed(model_seed)  # TensorFlow
    random.seed(model_seed)  # Python 内建 random 库

    # 初始化模型、优化器和损失函数
    model = LSPUFLRModel(phi_length=phi_length,n_xor=PUFNumber)


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

    log_filename = f"./Save/LSPUF/BreakingLSPUF/Log/FPGA_{PUFLength}_{PUFNumber}_{train_size / 1000:.0f}k_LSPUF_{batch_size}_{lr:.4f}_epc{epochs}_{load_seed}_{model_seed}.txt"


    with open(log_filename, "w") as f:
        for epoch in range(epochs):
            # 训练阶段
            model.trainable = True
            train_loss = 0.0

            #手动切 batch
            idx = np.arange(train_size)
            np.random.shuffle(idx)
            # X_train = X_train[idx]
            # Y_train = Y_train[idx]

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

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss += loss.numpy() * batch_x.shape[0]
                pred_labels = (preds > 0.5).numpy().astype(np.float32)
                correct_train += np.sum(pred_labels == batch_y)
                total += batch_y.shape[0]

            train_losses.append(train_loss / train_size)

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


    if best_val_acc > 0.97 or best_val_acc < 0.7:
        return model



    """ LSPUF : Refine the model found
    please refer to N. Wisiol, G. T. Becker, M. Margraf, T. A. Soroceanu, J. Tobisch, and B. Zengin,
     “Breaking the lightweight secure PUF: Understanding therelation of input transformations and machine learning resistance,”
      in Smart Card Research and Advanced Applications. Cham, Switzerland:Springer, 2020, pp. 40–54.>>
    """

    with open(f"{name}——{best_val_acc:.2%}{ext}","a+") as f:
        start_time = time.time()
        print("---------------------------------------------Start Swapping and Rotating the Weights-----------------------------------------------")
        f.write("---------------------------------------------Start Swapping and Rotating the Weights-----------------------------------------------\n")

        correlation_permutations = loadmat(
            'Datasets/correlation_permutations_lightweight_secure_%i_10.mat' % PUFLength
        )['shiftOverviewData'][:, :, 0].astype('int32')

        model_shift = LSPUFLRModel(n_xor=PUFNumber,phi_length=phi_length)
        model_shift.build(input_shape=(None, X_train.shape[1]))

        # 复制原模型所有参数
        model.trainable = True
        model_shift.trainable = True
        for var_src, var_dst in zip(model.trainable_variables, model_shift.trainable_variables):
            var_dst.assign(var_src)

        # find_high_accuracy_weight_permutations
        high_accuracy_permutations = []
        for permutation in list(itertools.permutations(range(PUFNumber)))[1:]:
            for i in range(PUFNumber):
                shifted_weight_i = tf.roll(model.puf_weights[i],shift=-correlation_permutations[i, permutation[i]],axis=0)
                model_shift.puf_weights[permutation[i]].assign(shifted_weight_i)

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

        # epochs = 500
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
                shifted_weight_i = tf.roll(model.puf_weights[i], shift=-correlation_permutations[i, permutation[i]],axis=0)
                model_shift.puf_weights[permutation[i]].assign(shifted_weight_i)

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
    PUFLength = 128
    filename = "./Datasets/LSPUF_128_4_CRPs.csv"
    proceed_filename = "." + filename.strip(".csv") + "_proceed.csv"
    data = np.loadtxt(filename, delimiter=",", dtype=int)
    assert PUFLength == data.shape[1] - 1
    N = PUFLength

    if not os.path.exists(proceed_filename):
        buffer_size = 1000000
        buffer = []
        with open(proceed_filename, "w", newline='') as f:
            writer = csv.writer(f)
            for i in range(data.shape[0]):
                C = data[i, :-1]
                challenge = C.tolist()
                R = data[i, -1]
                phi = np.zeros(shape=(PUFNumber, PUFLength + 1))

                for k in range(PUFNumber):
                    shift_challenge = APUF.shift(challenge, k)
                    line_challenge = [0] * N
                    line_challenge[N // 2] = shift_challenge[0]
                    for j in range(0, N, 2):
                        line_challenge[j // 2] = shift_challenge[j] ^ shift_challenge[j + 1]
                    for j in range(1, N - 1, 2):
                        line_challenge[(N + j + 1) // 2] = shift_challenge[j] ^ shift_challenge[j + 1]
                    phi[k] = APUF.transform(np.array(line_challenge))

                dataline = np.hstack((phi.flatten(), R)).astype(int)
                buffer.append(dataline)
                if len(buffer) >= buffer_size:
                    writer.writerows(buffer)
                    buffer = []  # 清空缓冲区

            if buffer:  # 写入剩余数据
                writer.writerows(buffer)
    # 训练参数
    train_ratio = 0.95
    n_outputs = (1,)
    epochs = 500
    save = True
    save_model = False

    method = "LR"

    num_sample = int(990 * 1000 / train_ratio)
    load_seeds = [58]
    model_seeds = [61]
    lrs = [30e-4]*20
    batch_sizes = [500 * 1000]



    for load_seed in load_seeds:
        for model_seed in model_seeds:
            for lr in lrs:
                for batch_size in batch_sizes:
                    np.random.seed()
                    load_seed = np.random.randint(100)
                    model_seed = np.random.randint(100)


                    print("-----------------------------------Start Attack-----------------------------------")
                    print(
                        f"PUFNumber: {PUFNumber} ,PUFLength: {PUFLength}  , num_sample:{num_sample}")
                    print(
                        f"method: {method}, lr: {lr}, batch_size: {batch_size}, epochs: {epochs} ,load_seed :{load_seed}, model_seed :{model_seed}")

                    AttackLSPUF(PUFNumber=PUFNumber, PUFLength=PUFLength,
                                filename=proceed_filename, num_sample=num_sample,
                                train_ratio=train_ratio,batch_size=batch_size, method=method, lr=lr, epochs=epochs, save=True,
                                load_seed=load_seed,model_seed=model_seed)

