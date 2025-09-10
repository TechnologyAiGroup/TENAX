import os
import time
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.decomposition import IncrementalPCA

from TRL import *
from PUF.XORAPUF import *
from Datasets.data import makeData


# ======================================================
# 0. CSV -> NPY 转换
# ======================================================
import numpy as np
import pandas as pd
import os

def csv_to_npy(csv_filename, npy_filename, dtype=np.int8, block_size=20_000_000, phi_length=129, DataSize=0):
    """
    CSV -> NPY int8，确保是标准 npy，不含 pickle。
    """
    if os.path.exists(npy_filename):
        print(f"已有NPY文件: {npy_filename}")
        return

    # 创建空 memmap
    shape = (DataSize, phi_length + 1)
    data_memmap = np.lib.format.open_memmap(npy_filename, dtype=dtype, mode='w+', shape=shape)

    written = 0
    for chunk in pd.read_csv(csv_filename, header=None, chunksize=block_size):
        arr = chunk.to_numpy(copy=False)  # 转 numpy
        arr = arr.astype(dtype, copy=False)  # 强制 int8
        rows = arr.shape[0]
        data_memmap[written:written+rows, :] = arr
        written += rows
        print(f"写入进度: {written}/{DataSize}")

    data_memmap.flush()
    del data_memmap
    print(f"转换完成: {npy_filename}, shape={shape}, dtype={dtype}")

# ======================================================
# 1. 从NPY大块读取 + 增量PCA
# ======================================================
def load_npy_in_blocks_for_pca(npy_filename, num_sample, phi_length, pca_components, block_size=20_000_000):
    ipca = IncrementalPCA(n_components=pca_components, batch_size=block_size)
    seen = 0

    data = np.load(npy_filename, mmap_mode="r")  # 内存映射，不会一次性加载
    # data = np.load(npy_filename, mmap_mode=None)  # 内存映射，一次性加载
    for start in range(0, num_sample, block_size):
        end = min(start + block_size, num_sample)
        X = data[start:end, :phi_length].astype(np.float32)
        X[X == -1] = 0.0
        ipca.partial_fit(X)
        seen += X.shape[0]
        print(f"PCA拟合进度: {seen}/{num_sample}")

    mu = ipca.mean_.astype(np.float32)
    coeff = ipca.components_.T.astype(np.float32)
    return mu, coeff


# ======================================================
# 2. 从NPY大块读取 + PCA变换 + batch生成（双层shuffle优化）
# ======================================================
def load_npy_in_blocks_for_training(npy_filename, num_sample, phi_length,
                                    mu, coeff, pca_components=None,
                                    block_size=20_000_000, batch_size=800_000,
                                    train_ratio=0.99, shuffle=True):
    train_size = int(num_sample * train_ratio)
    val_size = num_sample - train_size
    data = np.load(npy_filename, mmap_mode="r")  # 内存映射，避免一次性加载

    def data_generator(start, size, do_shuffle=shuffle):
        # --------- block 顺序 shuffle ---------
        block_starts = list(range(start, start + size, block_size))
        if do_shuffle:
            np.random.shuffle(block_starts)

        for block_start in block_starts:
            end = min(block_start + block_size, start + size)

            # --- 读 block ---
            block = data[block_start:end]
            # 避免重复拷贝，分开取 X, y
            X = block[:, :phi_length].astype(np.float32, copy=False)
            y = block[:, phi_length].astype(np.float32, copy=False).reshape(-1, 1)

            # --- block 内 shuffle ---
            if do_shuffle:
                idx = np.arange(X.shape[0])
                np.random.shuffle(idx)
                X = X[idx]
                y = y[idx]

            # --- 特征变换 ---
            if pca_components is not None and pca_components > 0:
                X_raw = X.copy()
                X[X == -1] = 0.0
                X_centered = X - mu[None, :]
                score = X_centered @ coeff
                X = np.concatenate([X_raw, score], axis=1)

            # --- 按 batch 输出 ---
            for i in range(0, X.shape[0], batch_size):
                xb = X[i:i+batch_size]
                yb = y[i:i+batch_size]
                yield xb, yb

    return (
        lambda: data_generator(0, train_size, do_shuffle=True),     # 训练集：打乱
        lambda: data_generator(train_size, val_size, do_shuffle=False),  # 验证集：不打乱
        train_size, val_size
    )




def AttackXPUF_by_C2D2_npy(PUFNumber=4, PUFLength=64, Noise=0.01, DataSize=80 * 1000,
                           filename="", num_sample=40 * 1000,
                           train_ratio=0.8, batch_size=10 * 1000,
                           lr=80e-4, n_outputs=(1,), epochs=1000,
                           save=True, l2_rate=0.0001,
                           pca_components=None, model_seed=42,
                           load_seed=42, data_seed=42):
    np.random.seed(data_seed)
    tf.random.set_seed(data_seed)
    random.seed(data_seed)

    # 构建数据集（若文件不存在则生成）
    XORAPUFSample = XORAPUF.randomSample(number=PUFNumber, length=PUFLength, noise_level=Noise)
    if not os.path.exists(filename):
        makeData(filename, DataSize, XORAPUFSample)

    phi_length = PUFLength + 1
    # 新增：CSV -> NPY
    npy_filename = filename.replace(".csv", ".npy")
    csv_to_npy(filename, npy_filename,block_size=20_000_000, phi_length=phi_length,DataSize=DataSize)


    # ---------------- PCA 拟合 ----------------
    if pca_components is not None and pca_components > 0:
        mu, coeff = load_npy_in_blocks_for_pca(
            npy_filename, num_sample, phi_length, pca_components, block_size=5_000_000
        )
    else:
        mu = np.zeros((phi_length,), dtype=np.float32)
        coeff = np.zeros((phi_length, 0), dtype=np.float32)

    # ---------------- 数据生成器 ----------------
    train_gen, val_gen, train_size, val_size = load_npy_in_blocks_for_training(
        npy_filename, num_sample, phi_length, mu, coeff,
        pca_components=pca_components,
        block_size=20_000_000, batch_size=batch_size, train_ratio=train_ratio
    )

    # ---------------- 定义模型 ----------------
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

    np.random.seed(model_seed)
    tf.random.set_seed(model_seed)
    random.seed(model_seed)
    model = LRModel(n_xor=PUFNumber)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    # ---------------- 训练循环 ----------------
    start_time = time.time()
    best_val_acc, best_train_acc = 0.0, 0.0
    log_filename = f"./Save/LR/Log/{PUFLength}_{PUFNumber}_{train_size / 1000:.0f}k_{Noise}_XORAPUF_{batch_size}_{lr:.4f}_epc{epochs}_l2={l2_rate}_{data_seed}_{load_seed}_{model_seed}.txt"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    with open(log_filename, "w") as f:
        for epoch in range(epochs):
            # ---- Train ----
            train_loss_sum = 0.0
            correct_train, total_train = 0, 0
            for batch_x, batch_y in train_gen():
                with tf.GradientTape() as tape:
                    preds = model(batch_x, training=True)
                    loss = loss_fn(batch_y, preds)
                    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if 'bias' not in v.name])
                    loss += l2_rate * l2_loss
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                train_loss_sum += float(loss.numpy()) * int(batch_x.shape[0])
                pred_labels = (preds > 0.5)
                correct_train += int(
                    tf.reduce_sum(tf.cast(tf.equal(pred_labels, tf.cast(batch_y > 0.5, tf.bool)), tf.int32)).numpy())
                total_train += int(batch_y.shape[0])

            epoch_train_loss = train_loss_sum / max(1, train_size)
            epoch_train_acc = correct_train / max(1, total_train)
            best_train_acc = max(best_train_acc, epoch_train_acc)

            # ---- Validation ----
            val_loss_sum = 0.0
            correct_val, total_val = 0, 0
            for val_x, val_y in val_gen():
                preds = model(val_x, training=False)
                loss = loss_fn(val_y, preds)
                val_loss_sum += float(loss.numpy()) * int(val_x.shape[0])
                pred_labels = (preds > 0.5)
                correct_val += int(
                    tf.reduce_sum(tf.cast(tf.equal(pred_labels, tf.cast(val_y > 0.5, tf.bool)), tf.int32)).numpy())
                total_val += int(val_y.shape[0])

            epoch_val_loss = val_loss_sum / max(1, val_size)
            epoch_val_acc = correct_val / max(1, total_val)
            best_val_acc = max(best_val_acc, epoch_val_acc)

            print(
                f"Epoch {epoch + 1}: Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2%}, Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2%}")
            f.write(
                f"Epoch {epoch + 1}: Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2%}, Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2%}\n")

        end_time = time.time()
        print(
            f"Runtime: {(end_time - start_time) / 60:.4f} min, Best Train Acc: {best_train_acc:.2%}, Best Val Acc: {best_val_acc:.2%}")
        f.write(f"Runtime: {(end_time - start_time) / 60:.4f} min\n")
        f.write(f"Best Train Acc: {best_train_acc:.2%}\n")
        f.write(f"Best Val Acc: {best_val_acc:.2%}\n")


if __name__ == "__main__":
    # 查看 GPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())

    # 配置参数
    PUFNumber = 7
    PUFLength = 128
    Noise = 0.01
    train_ratio = 0.99
    n_outputs = (1,)
    epochs = 50
    save = True

    DataSize = 410000 * 1000
    num_sample = int(400000 * 1000 / train_ratio)

    data_seeds = [42]
    load_seeds = [64]
    model_seeds = [99]
    l2_rates = [0]
    lrs = [60e-4]
    batch_sizes = [1600 * 1000]



    for data_seed in data_seeds:
        for load_seed in load_seeds:
            for model_seed in model_seeds:
                for lr in lrs:
                    for batch_size in batch_sizes:
                        for l2_rate in l2_rates:
                            np.random.seed()
                            load_seed = 41
                            model_seed =49

                            filename = f"./Datasets/{PUFLength}_{PUFNumber}_{DataSize / 1000:.0f}k_{Noise}_{data_seed}_XORAPUF.csv"

                            print("-----------------------------------Start Attack-----------------------------------")
                            print(
                                f"PUFNumber: {PUFNumber} ,PUFLength: {PUFLength} ,Noise: {Noise} , DataSize:{DataSize} , num_sample:{num_sample}")
                            print(
                                f"lr: {lr}, batch_size: {batch_size}, epochs: {epochs} , l2_rate : {l2_rate} , data_seed :{data_seed},load_seed :{load_seed}, model_seed :{model_seed}")

                            AttackXPUF_by_C2D2_npy(
                                PUFNumber=PUFNumber, PUFLength=PUFLength, Noise=Noise,
                                DataSize=DataSize, filename=filename, num_sample=num_sample,
                                train_ratio=train_ratio, batch_size=batch_size, lr=lr,
                                n_outputs=n_outputs, epochs=epochs, save=save, l2_rate=l2_rate,
                                pca_components=0,
                                load_seed=load_seed, model_seed=model_seed, data_seed=data_seed
                            )
