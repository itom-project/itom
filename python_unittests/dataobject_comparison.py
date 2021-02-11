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
        
        #until further notice, this test will fail, since itom <= 4.0.0 did not implement the
        #bool operator, such that dataObject == True always returned True, independent on the content.
        #This will be changed in the future and behaves than similar to np.ndarray: It will
        #raise a ValueError if the dataObject has more than one value, else the single value is
        #compared and the result of this single value comparison is returned.
        #see: https://bitbucket.org/itom/itom/issues/119/python-boolean-operator-of-dataobject
        dataObj = dataObject([2,3,4])
        with self.assertRaises(ValueError):
            result = not dataObj
        
    ##########################################################
    def test_bool(self):
        #until further notice, this test will fail, since itom <= 4.0.0 did not implement the
        #bool operator, such that dataObject == True always returned True, independent on the content.
        #This will be changed in the future and behaves than similar to np.ndarray: It will
        #raise a ValueError if the dataObject has more than one value, else the single value is
        #compared and the result of this single value comparison is returned.
        #see: https://bitbucket.org/itom/itom/issues/119/python-boolean-operator-of-dataobject
        
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

    def test_cmplx(self):
            # using complex comparators to force not all True / not all False
            # using randN and comparing, also using multiple dimensions to check implementation
            # using np.any() / np.all() to circumvent the current bool() 'error'
            
            dtypes = ['complex64','complex128']
            for dt in dtypes:
                zero = dataObject.zeros([2,3,4], dtype=dt)
                self.assertFalse(np.any(zero))
                
                one = dataObject.ones([2,3,4], dtype=dt)
                self.assertTrue(np.all(one))
                

                many = dataObject.randN([2,3,4], dtype=dt)
                # full DObject compare
                compmany = many.copy()
                self.assertTrue(np.all(many==compmany))
                compmany[0,0,0]*=5 #setting one value to be changed
                self.assertTrue(np.any(many!=compmany))
                
                #DObj to scalar
                self.assertFalse(np.any(many == 2+3j)) # randN -> all values should be < 1 -> should never be True
                self.assertTrue(np.all(many != 2+3j))
            
            #many = dataObject.randN([2,3,4]) #?
            #with self.assertRaises(ValueError):
            #    bool(many)
            
if __name__ == '__main__':
    unittest.main(module='dataobject_comparison', exit=False)