import unittest
from itom import dataObject
import numpy as np

class DataObjectUInt32Check(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_uint32(self):
        #it is not allowed to create an uint32 object or convert an existing object into this
        
        with self.assertRaises(TypeError):
            d = dataObject.randN([2,2], 'uint32')
        
        d = dataObject.randN([2,2],'float32')
        with self.assertRaises(TypeError):
            e = d.astype('uint32')
        
        a = np.array([2,2,2], 'uint32') #np.array for uint32 is allowed
        with self.assertRaises(RuntimeError):
            b = dataObject(a)


if __name__ == '__main__':
    unittest.main(module='dataobject_uint32_check', exit=False)