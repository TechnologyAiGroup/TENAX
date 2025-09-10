import numpy as np
from math import sqrt
from random import randint


class APUF:
    # Arbiter PUF
    # Attributes:
    #   length: the length of APUF
    #   noise: noise level of APUF, normal distributed
    #   weight: the delay of each stage, wiith the length of 64 + 1 or 128 + 1
    # Methods:
    #   __init__: init a object when use APUF(args)
    #   transform: trans challenge to phi
    #   getDelta: calc the time delay of the PUF with/without noise
    #   getResponse: get the response of the PUF with/without noise
    #   ramdomSample: inital a APUF class with random challenge and weight(normal distribution)

    def __init__(self, length, weight, noise_level=0):
        self.length = length
        self.weight = weight
        self.noise_level = noise_level
        assert self.weight.shape[-1] == self.length + 1, "Shape Error"

    @staticmethod
    def transform(challenge):
        phi = np.ones(shape=(challenge.shape[0] + 1,))
        for i in range(phi.shape[0] - 2, -1, -1):
            phi[i] = phi[i + 1] * (1 - 2 * challenge[i])
        return phi

    @staticmethod
    def shift(challenge: list, i) -> list:
        if i == 0:
            return challenge[-i:]
        else:
            return challenge[-i:] + challenge[:-i]

    def getDelta(self, phi, noisy=False):
        noise = np.zeros(shape=self.weight.shape)
        if noisy is True:
            noise = np.random.normal(0, self.noise_level, size=self.weight.shape)
        delta = np.sum(phi * (self.weight + noise), axis=-1, keepdims=True)
        return delta

    def getResponse(self, challenge=None, phi=None, noisy=False):
        if phi is None:
            assert challenge is not None, "Require challenge or phi"
            phi = APUF.transform(challenge)
        assert phi.shape[0] == self.length + 1, "Shape Error"
        delta = self.getDelta(phi, noisy)
        response = np.asarray([0])
        if delta < 0:
            response[0] = 1
        return response

    def randomSample(length=32, noise_level=0):
        weight = np.random.normal(0.1, 1, size=(length + 1,))
        return APUF(length, weight, noise_level)


'''
# TEST
length = 32
APUFSample = APUF.randomSample(length, noise_level=0.1)

print("Within different challenge")
for _ in range(5):
    challenge = np.asarray([randint(0, 1) for _ in range(length)])
    print("Response =", APUFSample.getResponse(challenge))

challenge = np.asarray([randint(0, 1) for _ in range(length)])
print("Within noise, where standard response is", APUFSample.getResponse(challenge))
for _ in range(10):
    print("Response =", APUFSample.getResponse(challenge, noisy=True))
#'''