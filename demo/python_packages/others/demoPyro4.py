"""Pyro4
========

This demo shows an example on how to the Pyro4 package.
"""

import Pyro4
import numpy as np
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPyro.png'


###############################################################################
# Define random generator class and register the host.
class RandGenerator:
    def getRand(self, x, y):
        return list(np.random.rand(y, x))


daemon = Pyro4.Daemon(host="129.69.65.61", port=12000)
uri = daemon.register(RandGenerator(), "123456")
print("uri=", uri)
daemon.requestLoop()

###############################################################################
# In Order to get it from another computer use following code:
import Pyro4

thing = Pyro4.Proxy("PYRO:123456@129.69.65.61:12000")
print(thing.getRand(42, 43))
