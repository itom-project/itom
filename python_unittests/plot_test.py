import unittest
from itom import plot
from itom import close
from itom import dataObject
import numpy as np

class PlotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_plotValues(self):
        res=True
        testObj=dataObject([1024,2048],'uint8')
        testObj[:,:]=255
        sumUp=np.sum(testObj,axis=0).astype('int32')
        i,h=plot(sumUp)
        displayed=h.call('getDisplayed')
        displayed=np.array(displayed)
        compare=displayed==sumUp
        close(i)
        if False in compare:
            res =False
        self.assertTrue(res)