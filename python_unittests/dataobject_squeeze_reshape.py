import unittest
from itom import dataObject

class DataObjectResize(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_reshape_uint16(self):
        obj = dataObject.randN([10,30,20,90],'uint16')
        obj_list = [i for i in obj]
        shapes = [[300,20,90],[2,30,5,20,90],[100,15,1,360]]
        for shape in shapes:
            b = obj.reshape(shape)
            self.assertEqual([i for i in b], obj_list)
            
    def test_reshape_float32(self):
        obj = dataObject.randN([10,45],'float32')
        obj_list = [i for i in obj]
        shapes = [[5,90],[5,2,45],[1,450],[450,1]]
        for shape in shapes:
            b = obj.reshape(shape)
            self.assertEqual([i for i in b], obj_list)
            
    def test_reshape_complex128(self):
        obj = dataObject.randN([8,10,20],'complex128')
        obj_list = [i for i in obj]
        shapes = [[4,2,5,2,20], [8,5,40],[4,5,4,20]]
        for shape in shapes:
            b = obj.reshape(shape)
            self.assertEqual([i for i in b], obj_list)
            
    def test_reshape_rgba32(self):
        obj = dataObject.randN([7,25,8,10,20],'rgba32')
        obj_list = [i for i in obj]
        shapes = [[175,8,10,20],[7,5,40,10,20],[7,25,80,20]]
        for shape in shapes:
            b = obj.reshape(shape)
            self.assertEqual([i for i in b], obj_list)
    
    def test_squeeze_continuous_obj(self):
        cdobj = dataObject([1,1,35,5,4],'float64',continuous=1)
        cdobj[:,:,:,:,:]=1
        cdobj[0,0,:,:,:]=2
        cdobj[0,0,0,:,:]=3
        a=cdobj[0,0,0,:,:].copy().squeeze()
        for i in a:
            self.assertEqual(i, 3)
            
    def test_reshape_continuous_obj(self):
        cdobj = dataObject([1,1,35,5,4],'float64',continuous=1)
        cdobj[:,:,:,:,:]=1
        a=cdobj.copy().reshape([35,5,4])
        for i in a:
            self.assertEqual(i, 1)
    
    def test_deepCopyPartial(self):
        obj = dataObject.zeros([2,4],'float32')
        obj2 = dataObject.ones([4,3],'float32')
        for i in range(4):
            obj2[i,:] = i
        obj[1,:] = obj2[:,1]
        
        for i in range(4):
            self.assertEqual(obj[1,i], obj2[i,1])

if __name__ == '__main__':
    unittest.main(module='dataobject_squeeze_reshape', exit=False)