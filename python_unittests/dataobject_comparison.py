import unittest
from itom import dataObject, rgba
import numpy as np
from numpy import testing as nptesting
from datetime import datetime, timedelta, timezone


class DataObjectComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    ##########################################################
    def test_invertComparison(self):
        npArray = np.ndarray([2, 3, 4])
        with self.assertRaises(ValueError):
            result = not npArray  # The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

        dataObj = dataObject([2, 3, 4])
        with self.assertRaises(ValueError):
            result = not dataObj

    ##########################################################
    def test_bool(self):
        dtypes = [
            "uint8",
            "int8",
            "uint16",
            "int16",
            "int32",
            "float32",
            "float64",
            "complex64",
            "complex128",
            "rgba32",
        ]
        for dt in dtypes:
            zero = dataObject.zeros([1], dtype=dt)
            self.assertFalse(bool(zero))

            one = dataObject.ones([1], dtype=dt)
            self.assertTrue(bool(one))

        nothing = dataObject()
        self.assertFalse(bool(nothing))

        many = dataObject.randN([2, 3, 4])
        with self.assertRaises(ValueError):
            bool(many)

    def test_comparison_equal(self):
        dtypes = [
            "uint8",
            # "int8",
            "uint16",
            "int16",
            "int32",
            "float32",
            "float64",
            "complex64",
            "complex128",
            "rgba32",
            "datetime",
            "timedelta",
        ]

        for dtype in dtypes:
            a = dataObject.zeros([100, 100], dtype)
            b = dataObject.zeros([100, 100], dtype)
            c = a == b
            nptesting.assert_array_equal(c, 255)

        number_values = [1, -2.75, 1 + 2j]

        # datetime
        a = dataObject([100, 100], "datetime")
        b = dataObject([100, 100], "datetime")
        dt = datetime(2000, 3, 12, 2, 3, 4, tzinfo=timezone(timedelta(0, 1800)))
        a[:, :] = dt
        b[:, :] = dt
        c = a == b
        nptesting.assert_array_equal(c, 255)

        c = a == dt
        nptesting.assert_array_equal(c, 255)

        for nv in number_values:
            with self.assertRaises(TypeError):
                a == nv

        # timedelta
        a = dataObject([100, 100], "timedelta")
        b = dataObject([100, 100], "timedelta")
        dt = timedelta(2000, 3, 12, 2, 3, 4)
        a[:, :] = dt
        b[:, :] = dt
        c = a == b
        nptesting.assert_array_equal(c, 255)

        c = a == dt
        nptesting.assert_array_equal(c, 255)

        for nv in number_values:
            with self.assertRaises(TypeError):
                a == nv

        # rgba
        a = dataObject([2, 1], "rgba32")
        b = dataObject([2, 1], "rgba32")
        a[0, 0] = rgba(10, 20, 30, 40)
        a[1, 0] = rgba(1, 20, 30, 4)
        b[0, 0] = rgba(10, 20, 30, 40)
        b[1, 0] = rgba(1, 2, 30, 4)
        c = a == b
        self.assertGreater(c[0, 0], 0)
        self.assertEqual(c[1, 0], 0)

        c = a == rgba(1, 20, 30, 4)
        self.assertEqual(c[0, 0], 0)
        self.assertEqual(c[1, 0], 255)

        for nv in number_values:
            with self.assertRaises(TypeError):
                a == nv

    def test_comparison_notequal(self):
        dtypes = [
            "uint8",
            # "int8",
            "uint16",
            "int16",
            "int32",
            "float32",
            "float64",
            "complex64",
            "complex128",
            "rgba32",
        ]

        for dtype in dtypes:
            a = dataObject.zeros([100, 100], dtype)
            b = dataObject.ones([100, 100], dtype)
            c = a != b
            nptesting.assert_array_equal(c, 255)

        number_values = [1, -2.75, 1 + 2j]

        # datetime
        a = dataObject([100, 100], "datetime")
        b = dataObject([100, 100], "datetime")
        c = dataObject([100, 100], "datetime")
        dt1 = datetime(2000, 3, 12, 2, 3, 4, tzinfo=timezone(timedelta(0, 1800)))
        dt2 = datetime(2000, 3, 12, 2, 3, 5, tzinfo=timezone(timedelta(0, 1800)))
        dt3 = datetime(2000, 3, 12, 2, 3, 4)
        a[:, :] = dt1
        b[:, :] = dt2
        c[:, :] = dt3
        d = a != b
        e = a != c
        d[0, 0]
        nptesting.assert_array_equal(d, 255)
        nptesting.assert_array_equal(e, 255)

        # datetime
        a = dataObject([100, 100], "datetime")
        b = dataObject([100, 100], "datetime")
        dt = datetime(2000, 3, 12, 2, 3, 4, tzinfo=timezone(timedelta(0, 1800)))
        a[:, :] = dt
        b[:, :] = dt
        c = a != b
        nptesting.assert_array_equal(c, 0)

        c = a != dt
        nptesting.assert_array_equal(c, 0)

        for nv in number_values:
            with self.assertRaises(TypeError):
                a != nv

        # timedelta
        a = dataObject([100, 100], "timedelta")
        b = dataObject([100, 100], "timedelta")
        dt = timedelta(2000, 3, 12, 2, 3, 4)
        a[:, :] = dt
        b[:, :] = dt
        c = a != b
        nptesting.assert_array_equal(c, 0)

        c = a != dt
        nptesting.assert_array_equal(c, 0)

        for nv in number_values:
            with self.assertRaises(TypeError):
                a != nv

        # rgba
        a = dataObject([2, 1], "rgba32")
        b = dataObject([2, 1], "rgba32")
        a[0, 0] = rgba(10, 20, 30, 40)
        a[1, 0] = rgba(1, 20, 30, 4)
        b[0, 0] = rgba(10, 20, 30, 40)
        b[1, 0] = rgba(1, 2, 30, 4)
        c = a != b
        self.assertEqual(c[0, 0], 0)
        self.assertGreater(c[1, 0], 0)

        c = a != rgba(200, 20, 30, 40)
        nptesting.assert_array_equal(c, 255)

        for nv in number_values:
            with self.assertRaises(TypeError):
                a != nv

    def test_greater(self):
        # datetime
        a = dataObject([100, 100], "datetime")
        b = dataObject([100, 100], "datetime")
        dt = datetime(2000, 3, 12, 2, 3, 4, tzinfo=timezone(timedelta(0, 1800)))
        a[:, :] = dt
        b[:, :] = dt
        c = a > b
        nptesting.assert_array_equal(c, 0)
        a[:, :] = datetime(2000, 3, 12, 2, 3, 5, tzinfo=timezone(timedelta(0, 1800)))
        c = a > b
        nptesting.assert_array_equal(c, 255)

        # timedelta
        a = dataObject([100, 100], "timedelta")
        b = dataObject([100, 100], "timedelta")
        dt = timedelta(2000, 3, 12, 2, 3, 4)
        a[:, :] = dt
        b[:, :] = dt
        c = a > b
        nptesting.assert_array_equal(c, 0)
        a[:, :] = timedelta(2000, 3, 12, 2, 3, 5)
        c = a > b
        nptesting.assert_array_equal(c, 255)

        # rgba
        a = dataObject([2, 1], "rgba32")
        b = dataObject([2, 1], "rgba32")
        a[0, 0] = rgba(10, 20, 30, 40)
        a[1, 0] = rgba(1, 20, 30, 4)
        b[0, 0] = rgba(10, 20, 30, 40)
        b[1, 0] = rgba(1, 2, 30, 4)
        with self.assertRaises(TypeError):
            c = a > b

    def test_smallerequal(self):
        # datetime
        a = dataObject([100, 100], "datetime")
        b = dataObject([100, 100], "datetime")
        dt = datetime(2000, 3, 12, 2, 3, 4, tzinfo=timezone(timedelta(0, 1800)))
        a[:, :] = dt
        b[:, :] = dt
        c = a <= b
        nptesting.assert_array_equal(c, 255)
        a[:, :] = datetime(2000, 3, 12, 2, 3, 5, tzinfo=timezone(timedelta(0, 1800)))
        c = a <= b
        nptesting.assert_array_equal(c, 0)

        # timedelta
        a = dataObject([100, 100], "timedelta")
        b = dataObject([100, 100], "timedelta")
        dt = timedelta(2000, 3, 12, 2, 3, 4)
        a[:, :] = dt
        b[:, :] = dt
        c = a <= b
        nptesting.assert_array_equal(c, 255)
        a[:, :] = timedelta(2000, 3, 12, 2, 3, 5)
        c = b <= a
        nptesting.assert_array_equal(c, 255)

        # rgba
        a = dataObject([2, 1], "rgba32")
        b = dataObject([2, 1], "rgba32")
        a[0, 0] = rgba(10, 20, 30, 40)
        a[1, 0] = rgba(1, 20, 30, 4)
        b[0, 0] = rgba(10, 20, 30, 40)
        b[1, 0] = rgba(1, 2, 30, 4)
        with self.assertRaises(TypeError):
            c = a <= b

    def test_complex(self):
        # using complex comparators to force not all True / not all False
        # using randN and comparing, also using multiple dimensions to check implementation
        # using np.any() / np.all() to circumvent the current bool() 'error'

        dtypes = ["complex64", "complex128"]
        for dt in dtypes:
            zero = dataObject.zeros([2, 3, 4], dtype=dt)
            self.assertFalse(np.any(zero))
            self.assertTrue(np.all(zero == zero))
            self.assertTrue(np.all((zero != zero) == 0))

            one = dataObject.ones([2, 3, 4], dtype=dt)
            self.assertTrue(np.all(one))
            self.assertTrue(np.all(one == one))
            self.assertTrue(np.all((one != one) == 0))

            many = dataObject.randN([2, 3, 4], dtype=dt)
            # full DObject compare
            compmany = many.copy()
            self.assertTrue(np.all(many == compmany))
            compmany[0, 0, 0] *= 5  # setting one value to be changed
            self.assertTrue(np.any(many != compmany))

            # DObj to scalar
            self.assertFalse(
                np.any(many == 2 + 3j)
            )  # randN -> all values should be < 1 -> should never be True
            self.assertTrue(np.all(many != 2 + 3j))

            # compare with int or float scalar
            self.assertTrue(np.all(zero == 0))
            self.assertTrue(np.all(zero == 0.0))
            self.assertTrue(np.all(one == 1))
            self.assertTrue(np.all(one == 1.0))
            self.assertTrue(np.all(one != 2))

            self.assertTrue(np.all(zero != one))
            self.assertFalse(np.any(zero == one))

        # it is not allowed to compare a real dataObject with a complex scalar
        dtypes = ["uint8", "int16", "int32", "float32", "float64"]
        for dt in dtypes:
            obj = dataObject.ones([2, 3, 4], dt)
            with self.assertRaises(TypeError):
                obj == 2.3j
            self.assertTrue(np.all(obj == 1))


if __name__ == "__main__":
    unittest.main(module="dataobject_comparison", exit=False)
