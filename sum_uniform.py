import math
import numpy as np
import matplotlib as mpl

mpl.use('Qt5Agg')
from matplotlib import pyplot as plt


# Returns a list of length n, with 1/n in each of the n slots around the center (+offset)
def uniform(n, k, offset_from_center):
    if n % 2 == 0:
        print("Error: N must be odd.")
        return
    list = [0] * n
    center = n // 2
    start = center + offset_from_center - k // 2
    end = start + k
    if start < 0 or end >= n:
        print("Out of index range")
        return
    else:
        for i in range(start, end):
            list[i] = 1 / k
    return list


def sum_of_uniforms(n, bits_per_draw, bits):
    if n % 2 == 0:
        print("Error: N must be odd.")
        return
    if bits_per_draw > bits:
        print("Error: Not enough bits.")
        return
    options = 2 << (bits_per_draw - 1)
    prob = uniform(n, options, 0)
    offset = 1
    for i in range(0, bits // bits_per_draw - 1):
        cur_prob = uniform(n, options, offset)
        offset = not offset
        prob = np.convolve(prob, cur_prob, 'same')

    count = np.count_nonzero(prob)
    plt.plot(prob, label=str(bits_per_draw) + " bits for a total of " + str(count), linestyle='None', marker='x')
    plt.legend()
    plt.show()


n = 1001
bits = 100
for k in range(1, 10):
    sum_of_uniforms(n, k, 100)
