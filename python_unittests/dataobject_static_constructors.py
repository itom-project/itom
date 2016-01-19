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


if __name__ == '__main__':
    unittest.main()