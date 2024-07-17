import numpy as np


def rl_deconv(img, num_iter=1, observer=None):
    out = np.flip(img)

    for i in range(num_iter):
        print("deconv")
        observer.progress(i + 1)
        observer.notify(f"Iteration{i}")

    return out
