import numpy as np
from .XORAPUF import XORAPUF

class IPUF:
    """
    Interpose PUF: 上层和下层均为 XORAPUF，
    下层 challenge 在 insert_index 位置插入上层单比特响应。
    """
    def __init__(self,
                 num_upper: int,
                 length: int,
                 num_lower: int,
                 insert_index: int,
                 noise_level: float = 0.0):
        # 参数检查
        assert isinstance(num_upper, int) and num_upper > 0, "num_upper must be positive int"
        assert isinstance(length, int) and length > 0, "length must be positive int"
        assert isinstance(num_lower, int) and num_lower > 0, "num_lower must be positive int"
        assert isinstance(insert_index, int), "insert_index must be int"
        assert 0 <= insert_index <= length, "insert_index must be in [0, length]"
        # 下层 challenge 长度应为 length + 1
        len_lower = length + 1

        self.num_upper = num_upper
        self.length = length
        self.num_lower = num_lower
        self.insert_index = insert_index
        self.noise_level = noise_level

        # 随机初始化上/下层 XORAPUF
        self.upper_puf = XORAPUF.randomSample(num_upper, length, noise_level)
        self.lower_puf = XORAPUF.randomSample(num_lower, len_lower, noise_level)

    def getResponse(self,
                    challenge: np.ndarray,
                    noisy: bool = False) -> np.ndarray:
        """
        :param challenge: 1D np.array of {0,1}, 长度 = length
        :param noisy: 是否添加噪声
        :return: np.array([0]) 或 np.array([1])
        """
        # 检查输入
        challenge = np.asarray(challenge, dtype=int)
        if challenge.ndim != 1 or challenge.shape[0] != self.length:
            raise ValueError(f"challenge must be 1D array of length {self.length}")

        # 上层 XORAPUF 输出 single-bit 响应
        upper_resp = self.upper_puf.getResponse(challenge, noisy=noisy)
        # 插入到下层 challenge
        lower_ch = np.concatenate([
            challenge[:self.insert_index],
            upper_resp,
            challenge[self.insert_index:]
        ])
        # 下层响应即最终响应
        final_resp = self.lower_puf.getResponse(lower_ch, noisy=noisy)
        return final_resp

    @staticmethod
    def randomSample(num_upper: int = 1,
                     length: int = 64,
                     num_lower: int = 4,
                     insert_index: int = 32,
                     noise_level: float = 0.01) -> "IPUF":
        """
        工厂方法：随机生成一个 IPUF 实例
        参数含义同 __init__。
        """
        return IPUF(num_upper, length, num_lower, insert_index, noise_level)
