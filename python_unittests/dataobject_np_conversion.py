import unittest
from itom import dataObject
import numpy as np

class DataObjectNpConversion(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_dataObject2NpArray(self):
        data = (1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19)
        a = dataObject([2,3,3],'uint8', data= data)
        b1 = np.array(a, copy = False)
        self.assertEqual(np.sum(data), np.sum(b1))
        b2 = np.array(a[:,1:3,:])
        result = 4+5+6+7+8+9+14+15+16+17+18+19
        self.assertEqual(np.sum(b2), result)
        sum = 0
        for i in a[:,1:3,:]:
            sum += i
        self.assertEqual(result, sum)


if __name__ == '__main__':
    unittest.main()