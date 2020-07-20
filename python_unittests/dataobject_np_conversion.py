import unittest
from itom import dataObject
import numpy as np

class DataObjectNpConversion(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_continuousDataObject2NpArray(self):
        data = (1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19)
        a = dataObject([2,3,3],'uint8', data= data, continuous = 1)
        b1 = np.array(a, copy = False)
        self.assertEqual(np.sum(data), np.sum(b1))
        b2 = np.array(a[:,1:3,:])
        result = 4+5+6+7+8+9+14+15+16+17+18+19
        self.assertEqual(np.sum(b2), result)
        sum = 0
        for i in a[:,1:3,:]:
            sum += i
        self.assertEqual(result, sum)
        self.assertIs(b1.base, a)
    
    def test_noncontinuousDataObject2NpArray(self):
            data = (1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19)
            a = dataObject([2,3,3],'uint8', data= data, continuous = 0)
            
            # a non-continuous dataObject cannot share its data with a np.ndarray.
            # Instead, an implicit deep-copy is done.
            b2 = np.array(a, copy = False)
            
            #make object continuous before going on
            b1 = np.array(a.makeContinuous(), copy = False)
            
            b2[0,0,0] = 100
            b1[0,0,0] = 99
            a[0,0,0] = 98
            self.assertEqual(b2[0,0,0], 100)
            self.assertEqual(b1[0,0,0], 99)
            self.assertEqual(a[0,0,0],98)
            self.assertIsNone(a.base)
            self.assertIsNotNone(b1.base)
            self.assertIsNotNone(b2.base)
            self.assertIsNot(b1.base, b2.base)
    
    def test_npArrayBool2DataObject(self):
        boolArray = np.array([[True,False],[False,True]], dtype='bool')
        dobj = dataObject(boolArray)
        self.assertEqual(dobj.dtype, 'uint8')
        self.assertEqual(dobj.shape, (2,2))
        self.assertEqual(dobj[0,0], 1)
        self.assertEqual(dobj[1,0], 0)
        self.assertEqual(dobj[0, 1], 0)
        self.assertEqual(dobj[1,1], 1)
        
    def test_npArray2dataObjectSameType(self):
        dtypes = ['uint8','int8','uint16','int16','int32','float32','float64','complex64','complex128']
        for dt in dtypes:
            nparray = np.ndarray([2,5,6],dtype=dt)
            
            # fill with some values
            for z in range(0, 2):
                for y in range(0, 5):
                    for x in range(0, 6):
                        nparray[z, y, x] = z * (-0.96) + y * x * 1.02
            
            dobj = dataObject(nparray)
            self.assertEqual(nparray.shape, dobj.shape)
            self.assertEqual(dobj.dtype, dt)
            diff = nparray - dobj
            self.assertEqual((np.abs(diff) < 0.00001).all(), True)
    
    def test_npArray2dataObjectDiffType(self):
        srcdtypes = ['uint8','int8','uint16','int16','int32','float32','float64']
        for dt in srcdtypes:
            for desttype in ['float32','complex128','int16']:
                nparray = np.ndarray([2,5,6],dtype=dt)
                
                # fill with some values
                for z in range(0, 2):
                        for y in range(0, 5):
                            for x in range(0, 6):
                                nparray[z, y, x] = z * (-0.96) + y * x * 1.02
                                       
                dobj = dataObject(nparray, dtype=desttype)
                self.assertEqual(nparray.shape, dobj.shape)
                diff = nparray.astype(desttype) - dobj
                self.assertEqual((np.abs(diff) < 0.00001).all(), True)
    
    def test_createEmptyDataObjectFromEmptyNpArray(self):
        '''this test reproduces the issue
        https://bitbucket.org/itom/itom/issues/114/systemerror-when-converting-an-empty
        '''
        dtypes = ['uint8','int8','uint16','int16','int32','float32','float64', 'complex64', 'complex128']
        for dtype in dtypes:
            nparray = np.array([], dtype=dtype)
            dobj = dataObject(nparray) #must pass properly
            self.assertEqual(len(dobj), 0)
            self.assertEqual(dobj.dtype, dtype)

if __name__ == '__main__':
    unittest.main(module='dataobject_np_conversion', exit=False)