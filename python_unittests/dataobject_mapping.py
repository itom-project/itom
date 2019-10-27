import unittest
from itom import dataObject
import numpy as np

class DataObjectMapping(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_mappingSetArrayLike(self):
        a = dataObject.zeros([10,10],'uint8')
        a[2:4,2:4] = [[11,12],[13,14]]
        self.assertEqual(np.sum(a), 11+12+13+14)
        
        a = dataObject.zeros([3,3],'uint8')
        a[0,:] = (100,101,102)
        self.assertEqual(np.sum(a), 100+101+102)
        
        a = dataObject.zeros([3,3],'uint8')
        a[:,0] = (100,101,102)
        self.assertEqual(np.sum(a), 100+101+102)
        
        a = dataObject.zeros([2,3,3],'uint8')
        a[1,:,2] = [100,101,102]
        self.assertEqual(np.sum(a), 100+101+102)
        
        a = dataObject.zeros([2,3,3],'uint8')
        a[:,0,:] = [[11,12,13],[110,120,130]]
        self.assertEqual(np.sum(a), 11+12+13+110+120+130)
        self.assertEqual(np.sum(a[0,0,:]), 11+12+13)
        self.assertEqual(np.sum(a[1,0,:]), 110+120+130)
        self.assertEqual(np.sum(a[:,1:3,:]), 0)


if __name__ == '__main__':
    unittest.main(module='dataobject_mapping', exit=False)