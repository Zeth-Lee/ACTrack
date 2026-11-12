# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved


import random

import numpy as np

__all__ = ["colormap", "random_color", "random_colors"]


_COLORS = (
    np.array(
        [

            1.000,
            1.000,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            1.000,
            0.498,
            0.000,
            0.498,
            1.000,
            0.000,
            0.000,
            1.000,
            0.498,
            1.000,
            0.000,
            0.498,
            0.498,
            0.000,
            1.000,
            0.753,
            1.000,
            0.000,
            1.000,
            0.753,
            0.000,
            0.000,
            1.000,
            0.753,
            0.753,
            0.000,
            1.000,
            1.000,
            0.000,
            0.753,
            1.000,
            0.251,
            0.000,
            0.251,
            1.000,
            0.000,
            0.000,
            1.000,
            0.251,
            0.251,
            0.000,
            1.000,
            1.000,
            0.000,
            0.251,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


def colormap(rgb=False, maximum=255):

    assert maximum in [255, 1], maximum
    c = _COLORS * maximum
    if not rgb:
        c = c[:, ::-1]
    return c


def random_color(rgb=False, maximum=255):

    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def random_colors(N, rgb=False, maximum=255):

    indices = random.sample(range(len(_COLORS)), N)
    ret = [_COLORS[i] * maximum for i in indices]
    if not rgb:
        ret = [x[::-1] for x in ret]
    return ret


if __name__ == "__main__":
    import cv2

    size = 100
    H, W = 10, 10
    canvas = np.random.rand(H * size, W * size, 3).astype("float32")
    for h in range(H):
        for w in range(W):
            idx = h * W + w
            if idx >= len(_COLORS):
                break
            canvas[h * size : (h + 1) * size, w * size : (w + 1) * size] = _COLORS[idx]
    cv2.imshow("a", canvas)
    cv2.waitKey(0)
