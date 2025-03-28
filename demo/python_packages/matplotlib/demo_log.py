"""Logarithmus
==============

"""

import numpy as np
from packaging import version

import matplotlib
import matplotlib.pyplot as plt

plt.figure()

plt.subplots_adjust(hspace=0.4)
t = np.arange(0.01, 20.0, 0.01)

# log y axis
plt.subplot(221)
plt.semilogy(t, np.exp(-t / 5.0))
plt.title("semilogy")
plt.grid(True)

# log x axis
plt.subplot(222)
plt.semilogx(t, np.sin(2 * np.pi * t))
plt.title("semilogx")
plt.grid(True)

# log x and y axis
plt.subplot(223)

if version.parse(matplotlib.__version__) < version.parse("3.3.0"):
    plt.loglog(t, 20 * np.exp(-t / 10.0), basex=2)
else:
    plt.loglog(t, 20 * np.exp(-t / 10.0), base=2)
plt.grid(True)
plt.title("loglog base 4 on x")

# with errorbars: clip non-positive values
ax = plt.subplot(224)

if version.parse(matplotlib.__version__) < version.parse("3.3.0"):
    ax.set_xscale("log", nonposx="clip")
    ax.set_yscale("log", nonposy="clip")
else:
    ax.set_xscale("log", nonpositive="clip")
    ax.set_yscale("log", nonpositive="clip")

x = 10.0 ** np.linspace(0.0, 2.0, 20)
y = x**2.0
plt.errorbar(x, y, xerr=0.1 * x, yerr=5.0 + 0.75 * y)
ax.set_ylim(ymin=0.1)
ax.set_title("Errorbars go negative")


plt.show()
