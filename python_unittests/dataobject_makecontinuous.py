import unittest
from itom import dataObject
import numpy as np

class DataObjectMakeContinuous(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_dataObjectMakeContinuous(self):
        data = (1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19)
        sum = np.sum(data)
        a = dataObject([2,3,3],'uint8', continuous = 0, data= data)
        ac = a.makeContinuous()
        for (i,j) in zip(a,ac):
            self.assertEqual(i,j)
            
        b = a[:,1:3,:]
        bc = b.makeContinuous()
        for (i,j) in zip(b,bc):
            self.assertEqual(i,j)
        
        b = a[:,1:3,1:3]
        bc = b.makeContinuous()
        for (i,j) in zip(b,bc):
            self.assertEqual(i,j)
            
        b = a[1:3,1:3,1:3]
        bc = b.makeContinuous()
        for (i,j) in zip(b,bc):
            self.assertEqual(i,j)


if __name__ == '__main__':
    unittest.main()