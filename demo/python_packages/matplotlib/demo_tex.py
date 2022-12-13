"""Tex
======

"""
# imports
import matplotlib.pyplot as plt
import numpy as np

# http://matplotlib.org/users/customizing.html
plt.rcParams["font.family"] = "serif"

def plot_figure(title: str, xLabel: str, yLabel: str, usetex: bool):
    fig, ax =plt.subplots(1, figsize=(6.29921, 5))  # create a new figure window
    # ax = plt.axes([0.1, 0.1, 0.8, 0.7])
    t = np.arange(0.0, 1.0 + 0.01, 0.01)
    s = np.cos(2 * 2 * np.pi * t) + 2

    ax.plot(t, s)  # plot line

    ax.set_xlabel(xLabel)  # x axis label
    ax.set_ylabel(yLabel)  # y axis label
    ax.set_title(title, fontsize=16, color="r")  # title

    ax.grid(True)  # create grid
    plt.tight_layout()
    plt.show()
    return

###############################################################################
# Generate without latex.usetext
title = r"TeX is Number $\sum_{n=1}^\infty \frac{-e^{i\pi}}{2^n}$!"
xLabel = r"time (s)"
yLabel = "velocity [°/sec]"
plot_figure(title, xLabel, yLabel, False)
