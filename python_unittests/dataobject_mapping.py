import unittest
from itom import dataObject, rgba
from datetime import datetime,timedelta
import numpy as np
from numpy import testing as nptesting

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

    def test_valueProperty(self):
        # integer
        values = [1, 7, 29, 113]

        for dtype in ['int8', 'uint8', 'int16', 'uint16', 'int32']:
            d = dataObject([1, 4], dtype, data = 0)
            nptesting.assert_array_equal(d.value, 0)
            d.value = values
            nptesting.assert_array_equal(d.value, values)

        # float
        values = [-45.5, 13.3, 0.0, 799.9994]
        
        for dtype in ['float64', 'float32']:
            d = dataObject([1, 4], dtype, data = 0)
            nptesting.assert_array_almost_equal(d.value, 0)
            d.value = values
            nptesting.assert_array_almost_equal(d.value, values, decimal=4)

        # complex
        values = [-45.5+2j, 13.3-13.3j, 0.0+0j, 799.9994+0.0001j]
        
        for dtype in ['complex128', 'complex64']:
            d = dataObject([1, 4], dtype, data = 0)
            nptesting.assert_array_almost_equal(d.value, 0)
            d.value = values
            nptesting.assert_array_almost_equal(d.value, values, decimal=4)

        # rgba32, currently only read-only
        values = [rgba(0,0,0), rgba(12,12,12,56), rgba(255,0,128,128), rgba(1,2,3,4)]
        d = dataObject([1, 4], "rgba32")
        for idx in range(len(values)):
            d[0,idx] = values[idx]
        for a, b in zip(d.value, values):
            self.assertEqual(a, b)

        # datetime, currently only read-only
        values = [datetime(2000,10,10), datetime(1910,1,1,14,13,12,2000), datetime(1899,12,31,23,59,59,99999), datetime(2099,12,31,23,59,59,99999)]
        d = dataObject([1, 4], "datetime")
        for idx in range(len(values)):
            d[0,idx] = values[idx]
        for a, b in zip(d.value, values):
            self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main(module='dataobject_mapping', exit=False)