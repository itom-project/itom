import unittest
from itom import dataObject

class DataObjectResize(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_reshape_uint16(self):
        obj = dataObject.randN([100,30,20,90],'uint16')
        obj_list = [i for i in obj]
        shapes = [[3000,20,90],[2,30,50,20,90],[100,15,10,360]]
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



if __name__ == '__main__':
    unittest.main()