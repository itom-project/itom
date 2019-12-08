import Pyro4
import numpy as np

class RandGenerator(object):
    def getRand(self, x, y):
        return list(np.random.rand(y,x))

def demo_pyro():
    # ------ normal code ------
    daemon = Pyro4.Daemon(host="129.69.65.61", port=12000)
    uri = daemon.register(RandGenerator(), "123456")
    print ("uri=",uri)
    #daemon.requestLoop()

    #in order to get it from another computer type:
    #import Pyro4
    #thing = Pyro4.Proxy("PYRO:123456@129.69.65.61:12000")
    #print thing.getRand(42,43)
if __name__ == "__main__":
    demo_pyro()