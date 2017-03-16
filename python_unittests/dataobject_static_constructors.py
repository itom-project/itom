import unittest
from itom import dataObject

class DataObjectStaticConstructors(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_default_types(self):
        '''all static constructors should return uint8 if not otherwise
        indicated (like the default constructor)'''
        obj = dataObject.zeros([10,10])
        self.assertEqual(obj.dtype, "uint8")
        
        obj = dataObject.ones([10,10])
        self.assertEqual(obj.dtype, "uint8")
        
        obj = dataObject.randN([10,10])
        self.assertEqual(obj.dtype, "uint8")
        
        obj = dataObject.rand([10,10])
        self.assertEqual(obj.dtype, "uint8")
        
        obj = dataObject.eye(5)
        self.assertEqual(obj.dtype, "uint8")

    def test_stack_dimensions(self):
        obj1=dataObject([2,2])
        obj2=dataObject([2,2])
        self.assertEqual(dataObject.dstack([obj1,obj2]).shape,(2,2,2))
        
        obj1=dataObject([1,3,1,1,2,2])
        obj2=dataObject([5,1,2,2])
        self.assertEqual(dataObject.dstack([obj1,obj2]).shape,(8,2,2))
        
        obj1=dataObject([2,3])
        obj2=dataObject([2,2])
        self.assertRaises(TypeError,dataObject.dstack,(obj1,obj2))

if __name__ == '__main__':
    unittest.main()