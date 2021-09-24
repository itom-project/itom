import unittest
from itom import dataObject
import numpy as np
from numpy import testing as nptesting


class DataObjectNpConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_continuousDataObject2NpArray(self):
        data = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19)
        a = dataObject([2, 3, 3], "uint8", data=data, continuous=1)
        b1 = np.array(a, copy=False)
        self.assertEqual(np.sum(data), np.sum(b1))
        b2 = np.array(a[:, 1:3, :])
        result = 4 + 5 + 6 + 7 + 8 + 9 + 14 + 15 + 16 + 17 + 18 + 19
        self.assertEqual(np.sum(b2), result)
        sum = 0
        for i in a[:, 1:3, :]:
            sum += i
        self.assertEqual(result, sum)
        self.assertIs(b1.base, a)

    def test_noncontinuousDataObject2NpArray(self):
        data = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19)
        a = dataObject([2, 3, 3], "uint8", data=data, continuous=0)

        # a non-continuous dataObject cannot share its data with a np.ndarray.
        # Instead, an implicit deep-copy is done.
        b2 = np.array(a, copy=False)

        # make object continuous before going on
        b1 = np.array(a.makeContinuous(), copy=False)

        b2[0, 0, 0] = 100
        b1[0, 0, 0] = 99
        a[0, 0, 0] = 98
        self.assertEqual(b2[0, 0, 0], 100)
        self.assertEqual(b1[0, 0, 0], 99)
        self.assertEqual(a[0, 0, 0], 98)
        self.assertIsNone(a.base)
        self.assertIsNotNone(b1.base)
        self.assertIsNotNone(b2.base)
        self.assertIsNot(b1.base, b2.base)

    def test_npArrayBool2DataObject(self):
        boolArray = np.array([[True, False], [False, True]], dtype="bool")
        dobj = dataObject(boolArray)
        self.assertEqual(dobj.dtype, "uint8")
        self.assertEqual(dobj.shape, (2, 2))
        self.assertEqual(dobj[0, 0], 1)
        self.assertEqual(dobj[1, 0], 0)
        self.assertEqual(dobj[0, 1], 0)
        self.assertEqual(dobj[1, 1], 1)

    def test_zeroStridesNpArray2DataObject(self):
        """If the last dimension of a np.array is 1, its internal step
        value is 0. Usually a value != 0 is required. This has to be considered
        by the itom conversion.

        This test is related to issue https://bitbucket.org/itom/itom/issues/176
        """
        arr = (np.random.rand(512, 512) * 10000).astype("int32")
        arr1 = arr[:, 0]  # 1 dim, strides  (2048,) --> ok
        darr1 = dataObject(arr1)
        nptesting.assert_array_equal(arr1.squeeze(), np.array(darr1).squeeze())
        arr2 = arr[:, 0:1]  # 2 dim, strides (2048, 4) --> ok
        darr2 = dataObject(arr2)
        nptesting.assert_array_equal(arr2.squeeze(), np.array(darr2).squeeze())

        x, y = np.ogrid[:512, :512]
        dx = dataObject(x)  # 2 dim, strides (4, 0) --> could be a problem
        nptesting.assert_array_equal(x.squeeze(), np.array(dx).squeeze())
        dy = dataObject(y)  # 2 dim, strides (0, 4) --> could be a problem
        nptesting.assert_array_equal(y.squeeze(), np.array(dy).squeeze())

        x, y, z = np.ogrid[:7, :5, :6]
        xbool = x.astype("bool")
        dx = dataObject(xbool)  # shape: (7,1,1), strides: (4,0,0)
        self.assertEqual(dx.dtype, "uint8")
        nptesting.assert_array_equal(xbool.squeeze(), np.array(dx).squeeze())
        x2 = x[::2, :, :]
        dx2 = dataObject(x2)
        nptesting.assert_array_equal(x2.squeeze(), np.array(dx2).squeeze())
        dy = dataObject(y)  # shape: (1,5,1), strides: (0,4,0)
        nptesting.assert_array_equal(y.squeeze(), np.array(dy).squeeze())
        dz = dataObject(z)  # shape: (1,1,6), strides: (0,0,4)
        nptesting.assert_array_equal(z.squeeze(), np.array(dz).squeeze())
        z2 = z[:, :, 1:3]
        dz2 = dataObject(z2)
        nptesting.assert_array_equal(z2.squeeze(), np.array(dz2).squeeze())

        arr_roi = arr[100:200, 50:70]
        darr_roi = dataObject(arr_roi)
        ll = darr_roi.locateROI()
        self.assertEqual(ll[0], [100, 512])
        self.assertEqual(ll[1], [0, 0])

        a = np.arange(3, dtype="float64")
        x = np.broadcast_to(a, (5, 3))  # strides: (0, 8)
        dx = dataObject(x)
        self.assertEqual(dx.shape, (5, 3))
        nptesting.assert_array_equal(
            dx, [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]
        )

    def test_npArray2dataObjectReadonly(self):
        x = np.array([[1, 2], [2, 3]])
        x.setflags(write=False)
        dx = dataObject(x)
        self.assertIsNot(x, dx.base)
        nptesting.assert_array_equal(dx, [[1, 2], [2, 3]])

        x = np.array([[True, False], [False, False]], dtype="bool")
        x.setflags(write=False)
        dx = dataObject(x)
        self.assertIsNot(x, dx.base)
        nptesting.assert_array_equal(dx, [[True, False], [False, False]])

    def test_npArray2dataObjectSameType(self):
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
        ]
        for dt in dtypes:
            nparray = np.ndarray([2, 5, 6], dtype=dt)

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
        srcdtypes = ["uint8", "int8", "uint16", "int16", "int32", "float32", "float64"]
        for dt in srcdtypes:
            for desttype in ["float32", "complex128", "int16"]:
                nparray = np.ndarray([2, 5, 6], dtype=dt)

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
        """this test reproduces the issue
        https://bitbucket.org/itom/itom/issues/114/systemerror-when-converting-an-empty
        """
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
        ]
        for dtype in dtypes:
            nparray = np.array([], dtype=dtype)
            dobj = dataObject(nparray)  # must pass properly
            self.assertEqual(len(dobj), 0)
            self.assertEqual(dobj.dtype, dtype)

    def test_convertNpArrayToDataObjectAndCheckForOriginNpTags(self):
        # dataObject.continuous is abused to trigger adding the _orgNp tags
        # If continuous is set to 255 the flags are added to a dataObject
        # which is converted from a numpy.ndarray
        npArray = np.array([True, False, True], "bool")
        dObj = dataObject(npArray, continuous=255)
        self.assertTrue("_orgNpShape" in dObj.tags)
        self.assertTrue("_orgNpDType" in dObj.tags)
        self.assertEqual(dObj.tags["_orgNpShape"], "[{}]".format(npArray.shape[0]))
        self.assertEqual(dObj.tags["_orgNpDType"], "bool")

        npArray = np.array([1, 2, 3, 4, 5, 6], "uint8")
        dObj = dataObject(npArray, continuous=255)
        self.assertTrue("_orgNpShape" in dObj.tags)
        self.assertTrue("_orgNpDType" in dObj.tags)
        self.assertEqual(dObj.tags["_orgNpShape"], "[{}]".format(npArray.shape[0]))
        self.assertEqual(dObj.tags["_orgNpDType"], "uint8")

        npArray = np.array([1, 2, 3, 4, 5, 6], "uint16")
        dObj = dataObject(npArray, continuous=255)
        self.assertTrue("_orgNpShape" in dObj.tags)
        self.assertTrue("_orgNpDType" in dObj.tags)
        self.assertEqual(dObj.tags["_orgNpShape"], "[{}]".format(npArray.shape[0]))
        self.assertEqual(dObj.tags["_orgNpDType"], "uint16")

        npArray = np.array([1, 2, 3, 4, 5, 6], "int8")
        dObj = dataObject(npArray, continuous=255)
        self.assertTrue("_orgNpShape" in dObj.tags)
        self.assertTrue("_orgNpDType" in dObj.tags)
        self.assertEqual(dObj.tags["_orgNpShape"], "[{}]".format(npArray.shape[0]))
        self.assertEqual(dObj.tags["_orgNpDType"], "int8")

        npArray = np.array([1, 2, 3, 4, 5, 6], "int16")
        dObj = dataObject(npArray, continuous=255)
        self.assertTrue("_orgNpShape" in dObj.tags)
        self.assertTrue("_orgNpDType" in dObj.tags)
        self.assertEqual(dObj.tags["_orgNpShape"], "[{}]".format(npArray.shape[0]))
        self.assertEqual(dObj.tags["_orgNpDType"], "int16")

        npArray = np.array([1, 2, 3, 4, 5, 6], "int32")
        dObj = dataObject(npArray, continuous=255)
        self.assertTrue("_orgNpShape" in dObj.tags)
        self.assertTrue("_orgNpDType" in dObj.tags)
        self.assertEqual(dObj.tags["_orgNpShape"], "[{}]".format(npArray.shape[0]))
        self.assertEqual(dObj.tags["_orgNpDType"], "long")

        npArray = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], "float")
        dObj = dataObject(npArray, continuous=255)
        self.assertTrue("_orgNpShape" in dObj.tags)
        self.assertTrue("_orgNpDType" in dObj.tags)
        self.assertEqual(dObj.tags["_orgNpShape"], "[{}]".format(npArray.shape[0]))
        self.assertEqual(dObj.tags["_orgNpDType"], "float64")

        npArray = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], "float32")
        dObj = dataObject(npArray, continuous=255)
        self.assertTrue("_orgNpShape" in dObj.tags)
        self.assertTrue("_orgNpDType" in dObj.tags)
        self.assertEqual(dObj.tags["_orgNpShape"], "[{}]".format(npArray.shape[0]))
        self.assertEqual(dObj.tags["_orgNpDType"], "float32")

        npArray = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], "complex64")
        dObj = dataObject(npArray, continuous=255)
        self.assertTrue("_orgNpShape" in dObj.tags)
        self.assertTrue("_orgNpDType" in dObj.tags)
        self.assertEqual(dObj.tags["_orgNpShape"], "[{}]".format(npArray.shape[0]))
        self.assertEqual(dObj.tags["_orgNpDType"], "complex64")

        npArray = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6], "complex128")
        dObj = dataObject(npArray, continuous=255)
        self.assertTrue("_orgNpShape" in dObj.tags)
        self.assertTrue("_orgNpDType" in dObj.tags)
        self.assertEqual(dObj.tags["_orgNpShape"], "[{}]".format(npArray.shape[0]))
        self.assertEqual(dObj.tags["_orgNpDType"], "complex128")

        
if __name__ == "__main__":
    unittest.main(module="dataobject_np_conversion", exit=False)
