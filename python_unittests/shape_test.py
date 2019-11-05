import unittest
from itom import dataObject
from itom import shape
import numpy as np

class ShapeTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass
        
    def test_rect_contains(self):
        '''tests if the contains method is correct for rectangle shapes'''
        shapes = (shape(shape.Rectangle,  (5, 3), (10, 5)), shape(shape.Rectangle,  (10,5), (5,3)))
        
        for item in shapes:
            pointsOut = [ 10.01, 10.01, 5,      6    , \
                                4,      7,       2.99, 2.99]
            pointsIn = [ 5.01,   9.9,    10,      5, \
                             3.01,   4.9,    3.2,      4.0]
            pointsOut2 = dataObject([2, 4], 'float64', data = pointsOut)
            pointsIn2 = dataObject([2, 4], 'float64', data = pointsIn)
            
            resultOut = item.contains( pointsOut2 )
            resultOutNp = item.contains( np.array(pointsOut2) )
            for i in range(pointsOut2.shape[1]):
                self.assertFalse( item.contains( [pointsOut2[0,i], pointsOut2[1,i]] ) )
                self.assertEqual( resultOut[0,i], 0 )
                self.assertEqual( resultOutNp[0,i], 0 )
            
            resultIn = item.contains( pointsIn2 )
            for i in range(pointsIn2.shape[1]):
                self.assertTrue( item.contains( [pointsIn2[0,i], pointsIn2[1,i]] ) )
                self.assertEqual( resultIn[0,i], 255 )
                
    def test_circle_contains(self):
        '''tests if the contains method is correct for circle shapes'''
        shapes = (shape(shape.Circle,  (-4, 3), 1.75), )
        
        for item in shapes:
            pointsOut = [ -4,    10.01, 5,      -5.76    , \
                                4.76,      7, 2.99, 3]
            pointsIn = [ -4,   -5.75,    -4 + 0.66,      -4, \
                             3,      3,         3 - 0.66,      3 + 1.75]
            pointsOut2 = dataObject([2, 4], 'float64', data = pointsOut)
            pointsIn2 = dataObject([2, 4], 'float64', data = pointsIn)
            
            resultOut = item.contains( pointsOut2 )
            resultOutNp = item.contains( np.array(pointsOut2) )
            for i in range(pointsOut2.shape[1]):
                self.assertFalse( item.contains( [pointsOut2[0,i], pointsOut2[1,i]] ) )
                self.assertEqual( resultOut[0,i], 0 )
                self.assertEqual( resultOutNp[0,i], 0 )
            
            resultIn = item.contains( pointsIn2 )
            for i in range(pointsIn2.shape[1]):
                self.assertTrue( item.contains( [pointsIn2[0,i], pointsIn2[1,i]] ) )
                self.assertEqual( resultIn[0,i], 255 )
    
    def test_ellipse_contains(self):
        '''tests if the contains method is correct for ellipse shapes'''
        shapes = (shape(shape.Ellipse,  (5, 3), (10, 5)), shape(shape.Ellipse,  (10,5), (5,3)))
        
        for item in shapes:
            pointsOut = [ 10.01, 10.01, 5,      5.5    , \
                                4,      7,       2.99, 4.7]
            pointsIn = [ 5.01,   7.5,    9.0,      5, \
                             4,        5 ,    3.2,      4.0]
            pointsOut2 = dataObject([2, 4], 'float64', data = pointsOut)
            pointsIn2 = dataObject([2, 4], 'float64', data = pointsIn)
            
            resultOut = item.contains( pointsOut2 )
            resultOutNp = item.contains( np.array(pointsOut2) )
            for i in range(pointsOut2.shape[1]):
                self.assertFalse( item.contains( [pointsOut2[0,i], pointsOut2[1,i]] ) )
                self.assertEqual( resultOut[0,i], 0 )
                self.assertEqual( resultOutNp[0,i], 0 )
            
            resultIn = item.contains( pointsIn2 )
            for i in range(pointsIn2.shape[1]):
                self.assertTrue( item.contains( [pointsIn2[0,i], pointsIn2[1,i]] ) )
                self.assertEqual( resultIn[0,i], 255 )
    
    def test_shape_contains_wrong_args(self):
        s = shape(shape.Point, (0,0))
        with self.assertRaises(RuntimeError):
            s.contains([3,4,5])
        with self.assertRaises(RuntimeError):
            s.contains(dataObject([1,7]))
        with self.assertRaises(RuntimeError):
            s.contains(dataObject([3,7]))
        with self.assertRaises(RuntimeError):
            s.contains(dataObject([2,7], 'complex64'))

    
    
if __name__ == '__main__':
    unittest.main(module='shape_test', exit=False)