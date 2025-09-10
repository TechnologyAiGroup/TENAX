import numpy as np
from math import sqrt
from random import randint
from .APUF import APUF

class lockFunc_V2:
    def call_lock(self, challenge, key):
        return [(c ^ k) for c, k in zip(challenge, key)]

# LSPUF
class LSPUF(APUF):
    def __init__(self, weight: np.ndarray, c_lock: lockFunc_V2, key: list, noise_level=0.0):
        length = weight.shape[-1] - 1
        super().__init__(length, weight, noise_level)
        self.c_lock = c_lock
        self.key = key

    def inputNetwork(self, challenge):
        if type(challenge) is not list:
            challenge = challenge.tolist()
        N = len(challenge)
        phi = np.zeros(shape=self.weight.shape)
        for i in range(self.weight.shape[0]):
            shift_challenge = APUF.shift(challenge, i)
            line_challenge = [0] * N
            line_challenge[N // 2] = shift_challenge[0]
            for j in range(0, N, 2):
                line_challenge[j // 2] = shift_challenge[j] ^ shift_challenge[j + 1]
            for j in range(1, N - 1, 2):
                line_challenge[(N + j + 1) // 2] = shift_challenge[j] ^ shift_challenge[j + 1]
            phi[i] = APUF.transform(np.array(line_challenge))
        return phi

    def getResponse(self, challenge=None, phi=None, noisy=False):
        if phi is None:
            assert challenge is not None, "Challenge must be provided"
            locked_challenge = self.c_lock.call_lock(challenge, self.key) if self.c_lock else challenge
            phi = self.inputNetwork(locked_challenge)
        assert phi.shape[-1] == self.weight.shape[-1] , "Shape Error"
        delta = self.getDelta(phi, noisy)
        outs = (delta >= 0).astype(int).flatten()
        response = np.asarray([np.bitwise_xor.reduce(outs)])
        return response

    @staticmethod
    def randomSample(Xnum=4, length=64, c_lock=None, key=None, sigma=1, alpha=0.0):
        weight = np.random.normal(0.1, sigma, size=(Xnum, length + 1))
        return LSPUF(weight, c_lock, key, noise_level=alpha * sigma)