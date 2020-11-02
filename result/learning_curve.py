import numpy as np
import matplotlib.pyplot as plt


with open('learning.txt', 'r') as f:
    results = []
    x = []
    for line in f.readlines():
        idx, seed1, seed2, seed3, seed4, seed5 = line.split()
        x.append(int(idx))
        results.append([float(seed1), float(seed2), float(seed3), float(seed4), float(seed5)])
    x = x[:100]
    result = results[:100]
    result = np.array(result)
    mean = np.mean(result, axis=1)
    std = np.std(result, axis=1)

    plt.plot(x, mean, '-', color='red', label="ppo reward")
    plt.fill_between(x, mean - std, mean + std, color='r', alpha=0.1)

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
