import unittest
from itom import dataObject
import numpy as np

class DataObjectScaleOffset(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_pix2phys(self):
        a = dataObject.zeros([60,100], 'float32')
        a.axisOffsets = (-30.0, 70.0)
        
        self.assertEqual(a.pixToPhys([0,0]), (30, -70))
        self.assertEqual(a.pixToPhys(0,0), 30.0)
        self.assertEqual(a.pixToPhys(0,1), -70.0)
        self.assertEqual(a.pixToPhys([0,0],[1,0]), (-70.0, 30.0))
        
        a.axisScales = (0.5, -0.3)
        self.assertEqual(a.pixToPhys([0,0]), (15, 21))
        self.assertEqual(a.pixToPhys(0,0), 15.0)
        self.assertEqual(a.pixToPhys(0,1), 21.0)
        self.assertEqual(a.pixToPhys([0,0],[1,0]), (21.0, 15.0))
        
        self.assertEqual(a.pixToPhys([10,20]), (20, 15))
        self.assertEqual(a.pixToPhys(10,0), 20.0)
        self.assertEqual(a.pixToPhys(20,1), 15.0)
        self.assertEqual(a.pixToPhys([20,10],[1,0]), (15.0, 20.0))
        
        #out of bounds does not raise a warning but calculates the true physical values
        self.assertEqual(a.pixToPhys([-5, -5]), (12.5, 22.5))

    def test_phys2pix(self):
        a = dataObject.zeros([60,100], 'float32')
        a.axisOffsets = (-30.0, 70.0)
        
        self.assertEqual(a.physToPix([30.0,-70.0]), (0,0))
        self.assertEqual(a.physToPix(30.0,0), 0)
        self.assertEqual(a.physToPix(-70.0,1), 0)
        self.assertEqual(a.physToPix([-70.0, 30.0],[1,0]), (0,0))
        
        a.axisScales = (0.5, -0.3)
        self.assertEqual(a.physToPix([15.0, 21.0]), (0,0))
        self.assertEqual(a.physToPix(15.0,0), 0)
        self.assertEqual(a.physToPix(21.0,1), 0)
        self.assertEqual(a.physToPix((21.0, 15.0),[1,0]), (0,0))
        
        self.assertEqual(a.physToPix([20.0,15.0]), (10, 20))
        self.assertEqual(a.physToPix(20.0,0), 10)
        self.assertEqual(a.physToPix(15.0,1), 20)
        self.assertEqual(a.physToPix([15.0, 20.0],[1,0]), (20,10))
        
        #out of bounds clips the value to the boundary pixel value (RuntimeWarning has to be raised
        with self.assertWarns(RuntimeWarning):
            ret1 = a.physToPix([12.5, 22.5])
        self.assertEqual(ret1, (0,0))
        
        with self.assertWarns(RuntimeWarning):
            ret2 = a.physToPix([45.5, -9.8])
        self.assertEqual(ret2, (59,99))


if __name__ == '__main__':
    unittest.main(module='dataobject_scale_offset', exit=False)