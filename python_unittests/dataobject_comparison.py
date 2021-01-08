import unittest
from itom import dataObject
import numpy as np

class DataObjectComparison(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass
    
    ##########################################################
    def test_invertComparison(self):
        npArray = np.ndarray([2,3,4])
        with self.assertRaises(ValueError):
            result = not npArray #The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        
        dataObj = dataObject([2,3,4])
        with self.assertRaises(ValueError):
            result = not dataObj
        
    ##########################################################
    def test_bool(self):
        dtypes = ['uint8','int8','uint16','int16','int32','float32','float64','complex64','complex128','rgba32']
        for dt in dtypes:
            zero = dataObject.zeros([1], dtype=dt)
            self.assertFalse(bool(zero))
            
            one = dataObject.ones([1], dtype=dt)
            self.assertTrue(bool(one))
            
        nothing = dataObject()
        self.assertFalse(bool(nothing))
        
        many = dataObject.randN([2,3,4])
        with self.assertRaises(ValueError):
            bool(many)

if __name__ == '__main__':
    unittest.main(module='dataobject_comparison', exit=False)