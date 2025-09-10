import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from TRL import *
from Datasets.data import *
from PUF.XORAPUF import *


def AttackXPUF_by_TRN_PCA_III(PUFNumber=4, PUFLength=64, Noise=0.01, DataSize=80 * 1000, filename="", num_sample=40 * 1000,
                      train_ratio=0.8, batch_size=10 * 1000, method="Ett", lr=80e-4, rank=[1,1,1,1,1], n_outputs=(1,),
                      epochs=1000, dropout = 0.0 ,save=True,save_model = False,threshhold = 0.97,l2_rate = 0.0001,pca_components=None,random_select_ratio = 0.9,model_seed=42,load_seed = 42,data_seed = 42):


    np.random.seed(data_seed)  # NumPy
    tf.random.set_seed(data_seed)  # TensorFlow
    random.seed(data_seed)  # Python 内建 random 库

    # 构建数据集
    XORAPUFSample = XORAPUF.randomSample(number=PUFNumber, length=PUFLength, noise_level=Noise)

    if not os.path.exists(filename):
        makeData(filename, DataSize, XORAPUFSample)

    # 划分数据集
    train_size = int(num_sample * train_ratio)
    val_size = num_sample - train_size
    phi_length = PUFLength + 1

    np.random.seed(load_seed)
    tf.random.set_seed(load_seed)
    random.seed(load_seed)
    X_train, Y_train, X_val, Y_val ,_ ,_,_= loadData_with_PCA_III(filename, num_sample, train_ratio, phi_length,random_select_ratio,pca_components=pca_components)

    # 定义模型结构
    class ETTModel(tf.keras.Model):
        def __init__(self, n_xor, dropout, rank, n_outputs):
            super(ETTModel, self).__init__()
            self.layer = ETTRegressionLayer(n_xor=n_xor, dropout=dropout, rank=rank, n_outputs=n_outputs)
            self.sigmoid = tf.keras.layers.Activation('sigmoid')

        def call(self, x):
            x = self.layer(x)
            return self.sigmoid(x)

    np.random.seed(model_seed)  # NumPy
    tf.random.set_seed(model_seed)  # TensorFlow
    random.seed(model_seed)  # Python 内建 random 库

    # 初始化模型、优化器和损失函数
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

    log_filename = f"./Save/{method}_PCA_III/Log/{PUFLength}_{PUFNumber}_{train_size / 1000:.0f}k_{Noise}_XORAPUF_{batch_size}_{lr:.4f}_{rank}_epc{epochs}_l2={l2_rate}_dp={dropout}_{data_seed}_{load_seed}_{model_seed}_{random_select_ratio}_.txt"
    figure_acc_path = f"./Save/{method}_PCA_III/Figures/Acc/{PUFLength}_{PUFNumber}_{train_size / 1000:.0f}k_{Noise}_XORAPUF_{batch_size}_{lr:.4f}_{rank}_epc{epochs}_l2={l2_rate}_dp={dropout}_{data_seed}_{load_seed}_{model_seed}_{random_select_ratio}_"
    figure_loss_path = f"./Save/{method}_PCA_III/Figures/Loss/{PUFLength}_{PUFNumber}_{train_size / 1000:.0f}k_{Noise}_XORAPUF_{batch_size}_{lr:.4f}_{rank}_epc{epochs}_l2={l2_rate}_dp={dropout}_{data_seed}_{load_seed}_{model_seed}_{random_select_ratio}_"
    loss_acc_path = f"./Save/{method}_PCA_III/Loss_Acc/{PUFLength}_{PUFNumber}_{train_size / 1000:.0f}k_{Noise}_XORAPUF_{batch_size}_{lr:.4f}_{rank}_epc{epochs}_l2={l2_rate}_dp={dropout}_{data_seed}_{load_seed}_{model_seed}_{random_select_ratio}_"

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
            if l2_loss.numpy() < 1e-1:
                print("L2_loss less than 1e-1, stop training.")
                if save == True:
                    f.write("L2_loss less than 1e-1, stop training.\n")
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
            if save_model == True and epoch > 300 and acc_val > best_val_acc and acc_val > threshhold:
                ModelName = f"./Save/{method}_PCA_III/Model/{PUFLength}_{PUFNumber}_{train_size / 1000:.0f}k_{Noise}_XORAPUF_{batch_size}_{lr:.4f}_{rank}_epc{epochs}_l2={l2_rate}_dp={dropout}_{data_seed}_{load_seed}_{model_seed}"
                model.save(ModelName)

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

    # # 绘制训练与验证损失曲线
    # plt.figure()
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training & Validation Loss')
    # plt.legend()
    # plt.grid(True)
    # if save:
    #     plt.savefig(figure_loss_path + "train_val_loss.png")
    #
    # plt.close()
    #
    # # 绘制验证准确率曲线
    # plt.figure()
    # plt.plot(train_acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Training & Validation Accuracy')
    # plt.legend()
    # plt.grid(True)
    # if save:
    #     plt.savefig(figure_acc_path + "train_val_acc.png")
    #
    # plt.close()
    #
    # # 保存结果为文本
    # if save:
    #     np.savetxt(loss_acc_path + "train_loss.csv", np.array(train_losses))
    #     np.savetxt(loss_acc_path + "train_acc.csv", np.array(train_acc))
    #     np.savetxt(loss_acc_path + "val_loss.csv", np.array(val_losses))
    #     np.savetxt(loss_acc_path + "val_acc.csv", np.array(val_acc))


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
            DataSize = 1000 * 1000
        elif PUFNumber == 7:
            DataSize = 2000 * 1000
        elif PUFNumber == 8:
            DataSize = 4000 * 1000

        if PUFNumber == 4:
            num_sample = int(40 * 1000 / 0.8)
        elif PUFNumber == 5:
            num_sample = int(80 * 1000 / 0.8)
        elif PUFNumber == 6:
            num_sample = int(320 * 1000 / 0.8)
        elif PUFNumber == 7:
            num_sample = int(560 * 1000 / 0.8)
        elif PUFNumber == 8:
            num_sample = int(2700 * 1000 / 0.8)

    elif PUFLength == 128:
        if PUFNumber == 4:
            DataSize = 300 * 1000
        elif PUFNumber == 5:
            DataSize = 1000 * 1000
        elif PUFNumber == 6:
            DataSize = 2000 * 1000
        elif PUFNumber == 7:
            DataSize = 4000 * 1000

        if PUFNumber == 4:
            num_sample = int(60 * 1000 / 0.8)
        elif PUFNumber == 5:
            num_sample = int(400 * 1000 / 0.8)
        elif PUFNumber == 6:
            num_sample = int(1200 * 1000 / 0.8)
        elif PUFNumber == 7:
            num_sample = int(2800 * 1000 / 0.8)

    # 训练参数
    train_ratio = 0.8
    n_outputs = (1,)
    epochs = 1000
    save = True
    save_model = False

    method = "Ett"

    # 4-xor search
    data_seeds = [42]
    load_seeds = [39]
    model_seeds = [88]
    random_select_ratios = [0.9]
    l2_rates = [1e-4]
    lrs = [60e-4]
    batch_sizes = [40 * 1000]
    rs = [1]
    dropouts = [0]

    for data_seed in data_seeds:
        for load_seed in load_seeds:
            for model_seed in model_seeds:
                for random_select_ratio in random_select_ratios:
                    for r in rs:
                        for lr in lrs:
                            for batch_size in batch_sizes:
                                for l2_rate in l2_rates:
                                    for dropout in dropouts:
                                        #np.random.seed()
                                        #load_seed = np.random.randint(100)
                                        #model_seed = np.random.randint(100)
                                        # TT_rank = [1, 1, 1, 1, r, 1, 1, 1, 1]
                                        # TT_rank = [1, 1, 1, 1, r, r, 1, 1, 1]
                                        # TT_rank = [1, 1, 1, 1, r, 1, 1, 1, 1]
                                        # TT_rank = [1, 1, 1, r, r, 1, 1, 1]
                                        # TT_rank = [1, 1, 1, r, 1, 1, 1]
                                        # TT_rank = [1, 1, 2, 2, 1, 1]
                                        # TT_rank = [1, 1, 1, 1, 1]
                                        TT_rank = [1] + [r] * (PUFNumber-1) + [1]

                                        if method == "Ett":
                                            rank = TT_rank
                                        else:
                                            rank = None
                                        filename = f"./Datasets/{PUFLength}_{PUFNumber}_{DataSize / 1000:.0f}k_{Noise}_{data_seed}_XORAPUF.csv"
                                        exist = False
                                        for fname in os.listdir(f"./Save/{method}_PCA_III/Log/"):
                                            if fname.startswith(
                                                    f"{PUFLength}_{PUFNumber}_{int(num_sample * train_ratio) / 1000:.0f}k_{Noise}_XORAPUF_{batch_size}_{lr:.4f}_{rank}_epc{epochs}_l2={l2_rate}_dp={dropout}_{data_seed}_{load_seed}_{model_seed}_{random_select_ratio}"):
                                                print("Already Trained")
                                                exist = True
                                                break
                                        if exist:
                                            continue

                                        print(
                                            "-----------------------------------Start Attack-----------------------------------")
                                        print(
                                            f"PUFNumber: {PUFNumber} ,PUFLength: {PUFLength} ,Noise: {Noise} , DataSize:{DataSize} , num_sample:{num_sample}")
                                        print(
                                            f"method: {method}, Rank: {rank}, lr: {lr}, batch_size: {batch_size}, epochs: {epochs} , l2_rate : {l2_rate} , dropout : {dropout} , data_seed :{data_seed},load_seed :{load_seed}, model_seed :{model_seed},random_select_ratio :{random_select_ratio}")

                                        AttackXPUF_by_TRN_PCA_III(PUFNumber=PUFNumber, PUFLength=PUFLength, Noise=Noise,
                                                                  DataSize=DataSize,
                                                                  filename=filename, num_sample=num_sample,
                                                                  train_ratio=train_ratio,
                                                                  batch_size=batch_size, method=method, lr=lr,
                                                                  rank=rank,
                                                                  n_outputs=n_outputs,
                                                                  epochs=epochs, dropout=dropout, save=save,
                                                                  l2_rate=l2_rate,
                                                                  pca_components=None,
                                                                  random_select_ratio=random_select_ratio,
                                                                  load_seed=load_seed,
                                                                  model_seed=model_seed, data_seed=data_seed)

