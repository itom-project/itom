import unittest
from itom import dataObject
import numpy as np
from numpy import testing as nptesting


class DataObjectMultiplication(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_matmul_operator_types(self):
        """Verify supported datatypes for matrix multiplication.
        
        Test that the matrix multiplication is only implemented for
        float32 and float64."""
        types_good = ["float32", "float64"]
        
        for typ in types_good:
            obj1 = dataObject.eye(3, typ)
            obj2 = dataObject.eye(3, typ)
            obj3 = obj1 @ obj2
            
            nptesting.assert_array_almost_equal(obj3, obj1)
        
        types_bad = ['uint8','int8','uint16','int16', 
                     'int32', 'complex64', 'complex128', 'rgba32']
        
        for typ in types_bad:
            obj1 = dataObject.eye(3, typ)
            obj2 = dataObject.eye(3, typ)
            
            with self.assertRaises(TypeError):
                obj1 @ obj2
    
    def test_matmul_operator_eye(self):
        """Verify the matmul of two eye matrix with different sizes and types."""
        types = ["float32", "float64"]
        
        for typ in types:
            for size in [1, 2, 3, 4, 5]:
                obj1 = dataObject.eye(size, typ)
                obj2 = dataObject.eye(size, typ)
                obj3 = obj1 @ obj2
                
                nptesting.assert_array_almost_equal(obj3, obj1)
    
    def test_matmul_operator(self):
        """Test some defaults math operations of the matmul operator."""
        types = ["float32", "float64"]
        
        for typ in types:
            a1 = dataObject([2, 3], dtype=typ, data=[-1, 2, 0.0001, 0.7, -3, 5])
            a2 = dataObject([3, 2], dtype=typ, data=[0.4, 100, -5e4, 22, 0.01, -0.01])
            a3 = dataObject([2, 2], dtype=typ, data=[1, 1, 0, 5])
            a4 = dataObject([3, 3], dtype=typ, data=
                            [-0.1, 0.2, -0.3, 0.4, 1, 2, -7, -5, 0])
            
            d1 = a1 @ a2
            n1 = np.array(a1) @ np.array(a2)
            nptesting.assert_array_almost_equal(d1, n1)
            
            d2 = a1 @ a4
            n2 = np.array(a1) @ np.array(a4)
            nptesting.assert_array_almost_equal(d2, n2)
            
            d3 = a2 @ a3
            n3 = np.array(a2) @ np.array(a3)
            nptesting.assert_array_almost_equal(d3, n3)
            
            d4 = a4 @ a2
            n4 = np.array(a4) @ np.array(a2)
            nptesting.assert_array_almost_equal(d4, n4)
            
    
    def test_matmul_operator_wrong(self):
        """Test some things that should fail with the matmul operator.
        """
        a = dataObject.eye(3, dtype='float32')
        a2 = dataObject.eye(4, dtype='float32')
        a3 = dataObject([3, 4], dtype='float64', data=list(range(0, 12)))
        
        with self.assertRaises(TypeError):
            b = a @ 2.0  # scalar multiplication not allowed
        
        with self.assertRaises(TypeError):
            b = 2 @ a  # scalar multiplication not allowed
        
        with self.assertRaises(TypeError):
            b = a @ np.eye(3)  # matmul with np.ndarray not allowed
        
        with self.assertRaises(TypeError):
            b = a3 @ a3  # shape inappropriate
        
        with self.assertRaises(TypeError):
            b = a2 @ a3  # shape inappropriate
    
    

if __name__ == '__main__':
    unittest.main(module='dataobject_multiplication', exit=False)