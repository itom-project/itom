import unittest
from itom import dataObject
import numpy as np


class DataObjectConstructors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    ##########################################################
    def test_constructor_simple(self):

        # DEFAULT TYPE: UINT8
        # test >= 2D
        shapes = [
            (2, 100),
            (1, 1000, 20),
            (2, 50, 31),
        ]
        for s in shapes:
            obj1 = dataObject(s)
            self.assertEqual(obj1.shape, s)
            obj1 = dataObject(list(s))
            self.assertEqual(obj1.shape, s)
            self.assertEqual(obj1.dtype, "uint8")

        # test 1D --> 1xN
        obj1 = dataObject(
            [
                100,
            ]
        )
        self.assertEqual(obj1.shape, (1, 100))
        self.assertEqual(obj1.dtype, "uint8")

        # GIVEN TYPE
        types = [
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

        for typ in types:
            for s in shapes:
                # kwds based
                obj1 = dataObject(s, dtype=typ)
                self.assertEqual(obj1.shape, s)
                self.assertEqual(obj1.dtype, typ)

                # non-kwds based
                obj1 = dataObject(s, typ)
                self.assertEqual(obj1.shape, s)
                self.assertEqual(obj1.dtype, typ)

    ##########################################################
    def test_continuous_constructor(self):

        # for 2d objects, there is no difference between continuous and non-continuous
        obj2dcont = dataObject([500, 512], continuous=1)
        obj2dnoncont = dataObject([500, 512], continuous=0)
        self.assertEqual(obj2dcont.continuous, obj2dnoncont.continuous)
        self.assertEqual(obj2dcont.continuous, True)

        # test the different for > 3 dims
        obj3dcont = dataObject([23, 47, 65], continuous=1)
        obj3dnoncont = dataObject([23, 47, 65], continuous=0)
        self.assertEqual(obj3dcont.continuous, True)
        self.assertEqual(obj3dnoncont.continuous, False)

    ##########################################################
    def test_copy_constructor_noncontinuous(self):
        a = dataObject([2, 100, 100], "uint16")
        b = dataObject(a)
        self.assertEqual(b.continuous, False)
        c = dataObject(a, continuous=True)
        self.assertEqual(c.continuous, True)
        cmp = a == c
        self.assertEqual(np.min(cmp), 255)

    ##########################################################
    def test_value_constructor(self):

        obj = dataObject([2, 4], "float32", data=[1, 2, 3, 4, 5, 6, 7, 8])
        for r in [0, 1]:
            for c in [0, 1, 2, 3]:
                self.assertEqual(obj[r, c], r * 4 + c + 1)
                self.assertEqual(type(obj[r, c]), float)

        obj = dataObject([2, 4], "int16", data=[1, 2, 3, 4, 5, 6, 7, 8])
        for r in [0, 1]:
            for c in [0, 1, 2, 3]:
                self.assertEqual(obj[r, c], r * 4 + c + 1)
                self.assertEqual(type(obj[r, c]), int)

        npvalues = np.array([1, 2, 3, 4, 5, 6])
        with self.assertRaises(TypeError):
            obj = dataObject([2, 3], "float32", data=npvalues)

        obj = dataObject([2, 3], "float32", data=npvalues.astype("float32"))
        self.assertEqual(obj[0, 0], 1)
        self.assertEqual(obj[1, 2], 6)

        obj = dataObject([2, 4], "float32", data=57)
        for r in [0, 1]:
            for c in [0, 1, 2, 3]:
                self.assertEqual(obj[r, c], 57.0)

        obj = dataObject([2, 4], "int8", data=4.8)
        for r in [0, 1]:
            for c in [0, 1, 2, 3]:
                self.assertEqual(obj[r, c], 5)

        with self.assertRaises(TypeError):
            obj = dataObject([2, 4], "int8", data=4 + 5j)

        obj = dataObject([2, 4], "complex64", data=4.2 + 5.8j)
        for r in [0, 1]:
            for c in [0, 1, 2, 3]:
                self.assertAlmostEqual(obj[r, c], 4.2 + 5.8j, places=5)

    ##########################################################
    def test_dtype_conversion_copy_constructor(self):
        obj1 = dataObject.randN([200, 200], "uint8")
        obj2 = dataObject(obj1)
        self.assertEqual(obj2.dtype, "uint8")
        obj3 = dataObject(obj1, dtype="uint8")
        self.assertEqual(obj3.dtype, "uint8")
        obj4 = dataObject(
            obj1, dtype="float32"
        )  # type cast to float32 should be done (like it is the case for np.ndarray)
        self.assertEqual(obj4.dtype, "float32")
        cmp = obj4 == obj1.astype("float32")
        self.assertEqual(np.min(cmp), 255)

    ##########################################################
    def test_copy_constructor(self):
        obj1 = dataObject.randN([200, 200], "uint8")
        obj2 = dataObject(obj1)
        self.assertEqual(obj2.dtype, "uint8")
        self.assertGreater(np.min(obj1 == obj2), 0)


if __name__ == "__main__":
    unittest.main(module="dataobject_constructors", exit=False)
