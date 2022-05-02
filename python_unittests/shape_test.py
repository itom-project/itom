import unittest
from itom import dataObject
from itom import shape
import numpy as np
from numpy import testing as nptesting


class ShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_rect_contains(self):
        """tests if the contains method is correct for rectangle shapes"""
        shapes = (
            shape(shape.Rectangle, (5, 3), (10, 5)),
            shape(shape.Rectangle, (10, 5), (5, 3)),
        )

        for item in shapes:
            pointsOut = [10.01, 10.01, 5, 6, 4, 7, 2.99, 2.99]
            pointsIn = [5.01, 9.9, 10, 5, 3.01, 4.9, 3.2, 4.0]
            pointsOut2 = dataObject([2, 4], "float64", data=pointsOut)
            pointsIn2 = dataObject([2, 4], "float64", data=pointsIn)

            resultOut = item.contains(pointsOut2)
            resultOutNp = item.contains(np.array(pointsOut2))
            for i in range(pointsOut2.shape[1]):
                self.assertFalse(item.contains([pointsOut2[0, i], pointsOut2[1, i]]))
                self.assertEqual(resultOut[0, i], 0)
                self.assertEqual(resultOutNp[0, i], 0)

            resultIn = item.contains(pointsIn2)
            for i in range(pointsIn2.shape[1]):
                self.assertTrue(item.contains([pointsIn2[0, i], pointsIn2[1, i]]))
                self.assertEqual(resultIn[0, i], 255)

    def test_circle_contains(self):
        """tests if the contains method is correct for circle shapes"""
        shapes = (shape(shape.Circle, (-4, 3), 1.75),)

        for item in shapes:
            pointsOut = [-4, 10.01, 5, -5.76, 4.76, 7, 2.99, 3]
            pointsIn = [-4, -5.75, -4 + 0.66, -4, 3, 3, 3 - 0.66, 3 + 1.75]
            pointsOut2 = dataObject([2, 4], "float64", data=pointsOut)
            pointsIn2 = dataObject([2, 4], "float64", data=pointsIn)

            resultOut = item.contains(pointsOut2)
            resultOutNp = item.contains(np.array(pointsOut2))
            for i in range(pointsOut2.shape[1]):
                self.assertFalse(item.contains([pointsOut2[0, i], pointsOut2[1, i]]))
                self.assertEqual(resultOut[0, i], 0)
                self.assertEqual(resultOutNp[0, i], 0)

            resultIn = item.contains(pointsIn2)
            for i in range(pointsIn2.shape[1]):
                self.assertTrue(item.contains([pointsIn2[0, i], pointsIn2[1, i]]))
                self.assertEqual(resultIn[0, i], 255)

    def test_ellipse_contains(self):
        """tests if the contains method is correct for ellipse shapes"""
        shapes = (
            shape(shape.Ellipse, (5, 3), (10, 5)),
            shape(shape.Ellipse, (10, 5), (5, 3)),
        )

        for item in shapes:
            pointsOut = [10.01, 10.01, 5, 5.5, 4, 7, 2.99, 4.7]
            pointsIn = [5.01, 7.5, 9.0, 5, 4, 5, 3.2, 4.0]
            pointsOut2 = dataObject([2, 4], "float64", data=pointsOut)
            pointsIn2 = dataObject([2, 4], "float64", data=pointsIn)

            resultOut = item.contains(pointsOut2)
            resultOutNp = item.contains(np.array(pointsOut2))
            for i in range(pointsOut2.shape[1]):
                self.assertFalse(item.contains([pointsOut2[0, i], pointsOut2[1, i]]))
                self.assertEqual(resultOut[0, i], 0)
                self.assertEqual(resultOutNp[0, i], 0)

            resultIn = item.contains(pointsIn2)
            for i in range(pointsIn2.shape[1]):
                self.assertTrue(item.contains([pointsIn2[0, i], pointsIn2[1, i]]))
                self.assertEqual(resultIn[0, i], 255)

    def test_shape_contains_wrong_args(self):
        s = shape(shape.Point, (0, 0))
        with self.assertRaises(RuntimeError):
            s.contains([3, 4, 5])
        with self.assertRaises(RuntimeError):
            s.contains(dataObject([1, 7]))
        with self.assertRaises(RuntimeError):
            s.contains(dataObject([3, 7]))
        with self.assertRaises(RuntimeError):
            s.contains(dataObject([2, 7], "complex64"))

    def test_create_point(self):
        center = (2.2, -7)
        pointShape = shape.createPoint(center, index=5, name="test")

        self.assertAlmostEqual(pointShape.center, center)
        self.assertEqual(pointShape.index, 5)
        self.assertEqual(pointShape.flags, 0)
        self.assertEqual(pointShape.name, "test")

        pointShape = shape.createPoint(point=(0, 0), flags=shape.MoveLock)
        self.assertEqual(pointShape.flags, shape.MoveLock)

        pointShape = shape(shape.Point, center, index=5, name="test")
        self.assertAlmostEqual(pointShape.center, center)
        self.assertEqual(pointShape.index, 5)
        self.assertEqual(pointShape.flags, 0)
        self.assertEqual(pointShape.name, "test")

    def test_create_mask_from_shapes(self):
        sourceObj = dataObject.zeros([220, 200], "uint8")

        # check rectangle
        rect = shape.createRectangle(center=(120, 80), size=(40, 20))
        maskRect = sourceObj.createMask(rect)
        maskRectDesired = sourceObj.copy()
        maskRectDesired[80 - 10 : 80 + 11, 120 - 20 : 120 + 21] = 255
        nptesting.assert_array_equal(maskRect, maskRectDesired)

        # check square
        square = shape.createSquare(center=(120, 80), sideLength=15)
        maskSquare = sourceObj.createMask(square)
        maskSquareDesired = sourceObj.copy()
        maskSquareDesired[80 - 8 : 80 + 9, 120 - 8 : 120 + 9] = 255
        nptesting.assert_array_equal(maskSquare, maskSquareDesired)
        self.assertEqual(maskSquare.shape, (220, 200))

        # check circle
        circle = shape.createCircle(center=(120, 80), radius=10)
        maskCircle = sourceObj.createMask(circle)
        self.assertEqual(maskCircle[80, 120], 255)
        self.assertEqual(maskCircle[0, 0], 0)
        self.assertEqual(maskCircle.shape, (220, 200))

        # check ellipse
        ellipse = shape.createEllipse(center=(120, 80), size=(40, 20))
        maskEllipse = sourceObj.createMask(ellipse)
        self.assertEqual(maskEllipse[80, 120], 255)
        self.assertEqual(maskEllipse[0, 0], 0)
        self.assertEqual(maskEllipse.shape, (220, 200))

        # check line (no mask)
        line = shape.createLine(point1=(10, 10), point2=(150, 150))
        maskLine = sourceObj.createMask(line)
        maskLineDesired = sourceObj.copy()
        nptesting.assert_array_equal(maskLine, sourceObj)
        self.assertEqual(maskLine.shape, (220, 200))

        # check multiple
        masks = sourceObj.createMask([rect, square, circle, ellipse, line])
        self.assertEqual(masks.shape, (220, 200))
        result = sourceObj.copy()
        result[maskRect > 0] = 255
        result[maskSquare > 0] = 255
        result[maskEllipse > 0] = 255
        result[maskLine > 0] = 255
        result[maskCircle > 0] = 255
        nptesting.assert_array_equal(masks, result)

    def test_ellipse_contour(self):
        def check_pts(contour: dataObject, cx: float, cy: float, a: float, b: float):
            for idx in range(contour.shape[1]):
                ptx = contour[0, idx] - cx
                pty = contour[1, idx] - cy
                diff = ptx * ptx / (a * a) + pty * pty / (b * b) - 1
                self.assertAlmostEqual(diff, 0.0, delta=0.1, msg="Idx %i" % idx)

        a = 20
        b = 10
        e1 = shape.createEllipse(center=(0, 0), size=(a * 2, b * 2))
        e1_contour = e1.contour()
        self.assertGreater(e1_contour.shape[1], 4)
        self.assertEqual(e1_contour.shape[0], 2)
        check_pts(e1_contour, 0, 0, a, b)

        e1 = shape.createEllipse(center=(0, 0), size=(-a * 2, -b * 2))
        e1_contour = e1.contour()
        self.assertGreater(e1_contour.shape[1], 4)
        self.assertEqual(e1_contour.shape[0], 2)
        check_pts(e1_contour, 0, 0, a, b)

        e1 = shape.createEllipse(center=(0, 0), size=(-a * 2, b * 2))
        e1_contour = e1.contour()
        self.assertGreater(e1_contour.shape[1], 4)
        self.assertEqual(e1_contour.shape[0], 2)
        check_pts(e1_contour, 0, 0, a, b)

    def test_center_point_and_area(self):
        def check(s: shape, cx, cy, area):
            s = s.copy()

            startAngle = 0.0

            for stepDeg in [0.0, -45, 25, 35, 7.2]:
                s.rotateDeg(stepDeg)
                startAngle += stepDeg
                self.assertAlmostEqual(s.angleDeg, startAngle)
                self.assertAlmostEqual(s.area, area)
                self.assertAlmostEqual(s.center[0], cx)
                self.assertAlmostEqual(s.center[1], cy)

        # check rectangle
        rect = shape.createRectangle(center=(120, 80), size=(40, 20))
        check(rect, 120, 80, 40 * 20)

        ellipse = shape.createRectangle(corner1=(120, 80), corner2=(80, 60))
        check(ellipse, 100, 70, 40 * 20)

        # check square
        square = shape.createSquare(center=(120, 80), sideLength=15)
        check(square, 120, 80, 15 * 15)

        # check circle
        circle = shape.createCircle(center=(120, 80), radius=10)
        check(circle, 120, 80, 10 * 10 * np.pi)

        # check ellipse
        ellipse = shape.createEllipse(center=(120, 80), size=(40, 20))
        check(ellipse, 120, 80, 20 * 10 * np.pi)

        ellipse = shape.createEllipse(corner1=(120, 80), corner2=(80, 60))
        check(ellipse, 100, 70, 20 * 10 * np.pi)

        # check line (no mask)
        line = shape.createLine(point1=(10, 10), point2=(150, 150))
        check(line, 80, 80, 0.0)


if __name__ == "__main__":
    unittest.main(module="shape_test", exit=False)
