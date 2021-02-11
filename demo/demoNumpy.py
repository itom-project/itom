"""
this demo shows an example on how to use numpy in itom
"""

import numpy as np

from rank_nullspace import rank, nullspace


def checkit(a):
    print("a:")
    print(a)
    r = rank(a)
    print("rank is", r)
    ns = nullspace(a)
    print("nullspace:")
    print(ns)
    if ns.size > 0:
        res = np.abs(np.dot(a, ns)).max()
        print("max residual is", res)


def demo_numpy():
    print("-" * 25)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    checkit(a)

    b = 2

    print("-" * 25)

    a = np.array([[0.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    checkit(a)

    print("-" * 25)

    a = np.array([[0.0, 1.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    checkit(a)

    print("-" * 25)

    a = np.array(
        [
            [1.0, 1.0j, 2.0 + 2.0j],
            [1.0j, -1.0, -2.0 + 2.0j],
            [0.5, 0.5j, 1.0 + 1.0j],
        ]
    )
    checkit(a)

    print("-" * 25)


if __name__ == "__main__":
    demo_numpy()
