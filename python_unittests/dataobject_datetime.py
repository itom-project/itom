import unittest
from itom import dataObject, rgba
import numpy as np
from numpy import testing as nptesting
from datetime import datetime, timedelta, timezone


class DataObjectDatetime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        tz = timezone(timedelta(0, 3600))

        self.dateObj = dataObject([2, 4, 3], "datetime")
        self.dt1 = datetime(2000, 8, 10, 4, 30, 10, 6754, tzinfo=tz)
        self.dt2 = datetime(2022, 1, 1, 0, 0, 0, 0)
        self.dateObj[0, :, :] = self.dt1
        self.dateObj[1, :, :] = self.dt2

        self.dateObj2 = self.dateObj.copy()
        # both rows are the same utc time
        self.dateObj2[0, :, :] = datetime(1980, 1, 1, 5)
        self.dateObj2[1, :, :] = datetime(1980, 1, 1, 6, tzinfo=tz)

        self.tdObj = dataObject([2, 4, 3], "timedelta")
        self.td1 = timedelta(-1, 5, 0, 0, 30, 6)
        self.td2 = timedelta(1, 1, 1, 1, 1, 1)
        self.tdObj[0, :, :] = self.td1
        self.tdObj[0, 0, 1] = timedelta(days=0, seconds=-24 * 3600 + 23405)
        self.tdObj[1, :, :] = self.td2

    def test_compare_operator(self):
        nptesting.assert_array_almost_equal(self.dateObj2 == self.dateObj, 0)
        nptesting.assert_array_almost_equal(self.dateObj2 != self.dateObj, 255)
        nptesting.assert_array_almost_equal(self.dateObj2 < self.dateObj, 255)
        nptesting.assert_array_almost_equal(self.dateObj2 <= self.dateObj, 255)
        nptesting.assert_array_almost_equal(self.dateObj2 > self.dateObj, 0)
        nptesting.assert_array_almost_equal(self.dateObj2 >= self.dateObj, 0)
        nptesting.assert_array_almost_equal(
            self.dateObj2[0, :, :] == self.dateObj2[1, :, :], 255
        )
        nptesting.assert_array_almost_equal(
            self.dateObj2[0, :, :] == self.dateObj2[1, :, :], 255
        )

        nptesting.assert_array_almost_equal(self.tdObj == self.tdObj, 255)
        nptesting.assert_array_almost_equal(self.tdObj != self.tdObj, 0)
        nptesting.assert_array_almost_equal(self.tdObj < self.tdObj, 0)
        nptesting.assert_array_almost_equal(self.tdObj <= self.tdObj, 255)
        nptesting.assert_array_almost_equal(self.tdObj > self.tdObj, 0)
        nptesting.assert_array_almost_equal(self.tdObj >= self.tdObj, 255)

    def test_operator_add_dataObjects(self):
        # test 1a. it is not allowed to add any datatype to a datetime
        # object besides a timedelta
        blacklist = [
            "uint8",
            "int8",
            "uint16",
            "int16",
            "int32",
            "float32",
            "float64",
            "complex64",
            "complex128",
            "datetime",
            "rgba32",
        ]

        for b in blacklist:
            obj = dataObject([2, 4, 3], b)
            with self.assertRaises(TypeError):
                self.dateObj + obj

            with self.assertRaises(TypeError):
                obj2 = self.dateObj.copy()
                obj2 += obj

        # test 1b. it is not allowed to add any datatype to a timedelta
        # object besides another timedelta or datetime
        blacklist = [
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

        for b in blacklist:
            obj1 = dataObject([2, 4, 3], "timedelta")
            obj2 = dataObject([2, 4, 3], b)
            with self.assertRaises(TypeError):
                obj1 + obj2

            with self.assertRaises(TypeError):
                obj2 = self.tdObj.copy()
                obj2 += obj

        ## OUT-OF-PLACE ADD

        # test 2: adding a datetime and timedelta is allowed
        result = self.dateObj + self.tdObj

        self.assertEqual(result.dtype, "datetime")

        for val in result[0, :, :]:
            self.assertAlmostEqual(val, self.dt1 + self.td1)

        for val in result[1, :, :]:
            self.assertAlmostEqual(val, self.dt2 + self.td2)

        # test 3: adding a timedelta and datetime is allowed
        result = self.tdObj + self.dateObj

        self.assertEqual(result.dtype, "datetime")

        for val in result[0, :, :]:
            self.assertAlmostEqual(val, self.dt1 + self.td1)

        for val in result[1, :, :]:
            self.assertAlmostEqual(val, self.dt2 + self.td2)

        # test 4: adding a timedelta and timedelta is allowed
        result = self.tdObj + self.tdObj

        self.assertEqual(result.dtype, "timedelta")

        for val in result[0, :, :]:
            self.assertAlmostEqual(val, self.td1 + self.td1)

        for val in result[1, :, :]:
            self.assertAlmostEqual(val, self.td2 + self.td2)

        ## IN-PLACE ADD

        # test 2: adding a timedelta to adatetime is allowed
        result = self.dateObj.copy()
        result += self.tdObj

        self.assertEqual(result.dtype, "datetime")

        for val in result[0, :, :]:
            self.assertAlmostEqual(val, self.dt1 + self.td1)

        for val in result[1, :, :]:
            self.assertAlmostEqual(val, self.dt2 + self.td2)

        # test 3: adding a datetime to a timedelta is not allowed
        result = self.tdObj.copy()

        with self.assertRaises(TypeError):
            result += self.dateObj

        # test 4: adding a timedelta to a timedelta is allowed
        result = self.tdObj.copy()
        result += self.tdObj

        self.assertEqual(result.dtype, "timedelta")

        for val in result[0, :, :]:
            self.assertAlmostEqual(val, self.td1 + self.td1)

        for val in result[1, :, :]:
            self.assertAlmostEqual(val, self.td2 + self.td2)

    def test_operator_subtract_dataObjects(self):
        # test 1. it is not allowed to subtract any datatype from a datetime
        # object besides a timedelta and another datetime
        blacklist = [
            "uint8",
            "int8",
            "uint16",
            "int16",
            "int32",
            "float32",
            "float64",
            "complex64",
            "complex128",
            "datetime",
            "rgba32",
        ]

        for b in blacklist:
            obj = dataObject([2, 4, 3], b)
            with self.assertRaises(TypeError):
                self.tdObj - obj

            with self.assertRaises(TypeError):
                obj2 = self.tdObj.copy()
                obj2 -= obj

        ## OUT-OF-PLACE SUBTRACT

        # test 2: subtracting a timedelta from a datetime is allowed
        result = self.dateObj - self.tdObj

        self.assertEqual(result.dtype, "datetime")

        for val in result[0, :, :]:
            self.assertAlmostEqual(val, self.dt1 - self.td1)

        for val in result[1, :, :]:
            self.assertAlmostEqual(val, self.dt2 - self.td2)

        # test 3: subtracting a datetime from a timedelta is not allowed
        with self.assertRaises(TypeError):
            result = self.tdObj - self.dateObj

        # test 4: subtract a timedelta from a timedelta is allowed
        result = self.tdObj - self.tdObj

        self.assertEqual(result.dtype, "timedelta")

        for val in result[0, :, :]:
            self.assertAlmostEqual(val, self.td1 - self.td1)

        for val in result[1, :, :]:
            self.assertAlmostEqual(val, self.td2 - self.td2)

        # test 5: subtracting a datetime from a datetime is allowed
        result = self.dateObj - self.dateObj
        self.assertAlmostEqual(result[0, 0, 0], self.dt1 - self.dt1)

        ## IN-PLACE SUBTRACT

        # test 2: subtracting a timedelta to a datetime is allowed
        result = self.dateObj.copy()
        result += self.tdObj

        self.assertEqual(result.dtype, "datetime")

        for val in result[0, :, :]:
            self.assertAlmostEqual(val, self.dt1 + self.td1)

        for val in result[1, :, :]:
            self.assertAlmostEqual(val, self.dt2 + self.td2)

        # test 3: adding a datetime to a timedelta is not allowed
        result = self.tdObj.copy()

        with self.assertRaises(TypeError):
            result += self.dateObj

        # test 4: adding a timedelta to a timedelta is allowed
        result = self.tdObj.copy()
        result += self.tdObj

        self.assertEqual(result.dtype, "timedelta")

        for val in result[0, :, :]:
            self.assertAlmostEqual(val, self.td1 + self.td1)

        for val in result[1, :, :]:
            self.assertAlmostEqual(val, self.td2 + self.td2)

    def test_operator_negate(self):
        tdneg = -self.tdObj
        for t1, t2 in zip(tdneg, self.tdObj):
            self.assertAlmostEqual(t1, -t2)

        with self.assertRaises(TypeError):
            # unary - operator not supported for datetime
            -self.dateObj

    def test_operator_multdiv(self):
        with self.assertRaises(TypeError):
            self.dateObj.mul(self.dateObj)

        with self.assertRaises(TypeError):
            self.dateObj * 2

        with self.assertRaises(TypeError):
            self.dateObj * 2.5

        with self.assertRaises(TypeError):
            self.dateObj * (2j + 3)

        with self.assertRaises(TypeError):
            self.dateObj *= 2

        with self.assertRaises(TypeError):
            self.dateObj *= 2.5

        with self.assertRaises(TypeError):
            self.dateObj *= 2j + 3

        with self.assertRaises(TypeError):
            self.dateObj.mul(self.tdObj)

        with self.assertRaises(TypeError):
            self.dateObj.div(self.dateObj)

        with self.assertRaises(TypeError):
            self.dateObj.div(self.tdObj)

        with self.assertRaises(TypeError):
            self.tdObj.mul(self.dateObj)

        with self.assertRaises(TypeError):
            self.tdObj.mul(self.tdObj)

        with self.assertRaises(TypeError):
            self.tdObj.div(self.dateObj)

        with self.assertRaises(TypeError):
            self.tdObj.div(self.tdObj)

        tdmult = self.tdObj * 2
        for t1, t2 in zip(tdmult, self.tdObj):
            self.assertEqual(t1, t2 * 2)

        tdmult = self.tdObj * 2.5
        for t1, t2 in zip(tdmult, self.tdObj):
            self.assertAlmostEqual(t1, t2 * 2.5)

        with self.assertRaises(TypeError):
            self.tdObj * (2j + 3)

        tdmult = self.tdObj.copy()
        tdmult *= 2
        for t1, t2 in zip(tdmult, self.tdObj):
            self.assertEqual(t1, t2 * 2)

        tdmult = self.tdObj.copy()
        tdmult *= 2.5
        for t1, t2 in zip(tdmult, self.tdObj):
            self.assertAlmostEqual(t1, t2 * 2.5)

        with self.assertRaises(TypeError):
            self.tdObj *= 2j + 3

    def test_abs(self):
        with self.assertRaises(TypeError):
            abs(self.dateObj)

        tdabs = abs(self.tdObj)

        for t1, t2 in zip(tdabs, self.tdObj):
            self.assertEqual(t1, abs(t2))

    def test_negate(self):
        with self.assertRaises(TypeError):
            ~(self.dateObj)

        with self.assertRaises(TypeError):
            ~(self.tdObj)

    def test_shift(self):
        with self.assertRaises(TypeError):
            self.dateObj << 2

        with self.assertRaises(TypeError):
            self.dateObj >> 2

        with self.assertRaises(TypeError):
            self.dateObj <<= 2

        with self.assertRaises(TypeError):
            self.dateObj >>= 2

        with self.assertRaises(TypeError):
            self.tdObj << 2

        with self.assertRaises(TypeError):
            self.tdObj >> 2

        with self.assertRaises(TypeError):
            self.tdObj <<= 2

        with self.assertRaises(TypeError):
            self.tdObj >>= 2

    def test_bitops(self):
        for obj in [self.dateObj, self.tdObj]:
            obj2 = obj.copy()

            with self.assertRaises(TypeError):
                obj ^ obj2

            with self.assertRaises(TypeError):
                obj ^= obj2

            with self.assertRaises(TypeError):
                obj | obj2

            with self.assertRaises(TypeError):
                obj |= obj2

            with self.assertRaises(TypeError):
                obj & obj2

            with self.assertRaises(TypeError):
                obj &= obj2

    def test_complex_operators(self):
        for obj in [self.dateObj, self.tdObj]:
            with self.assertRaises(TypeError):
                obj.imag

            with self.assertRaises(TypeError):
                obj.imag = 2

            with self.assertRaises(TypeError):
                obj.real

            with self.assertRaises(TypeError):
                obj.real = 5

            with self.assertRaises(TypeError):
                obj.arg()

            with self.assertRaises(TypeError):
                obj.adj()

            with self.assertRaises(TypeError):
                obj.adjugate()

            with self.assertRaises(TypeError):
                obj.conj()

            with self.assertRaises(TypeError):
                obj.conjugate()

    def test_npdatetime64_to_dataObject(self):
        # timebase us
        t = np.arange(datetime(1985, 7, 1), datetime(2003, 7, 1), timedelta(days=1))
        t[20] = datetime(2010, 6, 7, 16, 23, 59)
        dObj = dataObject(t)

        for _t, _d in zip(t, dObj):
            self.assertTrue(_t == np.datetime64(_d))

        t = np.array([np.datetime64(1, "Y"), np.datetime64(-5, "Y")])
        dObj = dataObject(t)

        for _t, _d in zip(t, dObj):
            self.assertTrue(_t == np.datetime64(_d))

        t = np.array([np.datetime64(1, "M"), np.datetime64(-5, "M")])
        dObj = dataObject(t)

        for _t, _d in zip(t, dObj):
            self.assertTrue(_t == np.datetime64(_d))

        t = np.array([np.datetime64(1, "D"), np.datetime64(-5, "D")])
        dObj = dataObject(t)

        for _t, _d in zip(t, dObj):
            self.assertTrue(_t == np.datetime64(_d))

        t = np.array([np.datetime64(1, "W"), np.datetime64(-5, "W")])
        dObj = dataObject(t)

        for _t, _d in zip(t, dObj):
            self.assertTrue(_t == np.datetime64(_d))

        t = np.array([np.datetime64(1, "h"), np.datetime64(-5, "h")])
        dObj = dataObject(t)

        for _t, _d in zip(t, dObj):
            self.assertTrue(_t == np.datetime64(_d))

        t = np.array([np.datetime64(1, "us"), np.datetime64(-5, "us")])
        dObj = dataObject(t)

        for _t, _d in zip(t, dObj):
            self.assertTrue(_t == np.datetime64(_d))

        t = np.array([np.datetime64(1, "s"), np.datetime64(-5, "s")])
        dObj = dataObject(t)

        for _t, _d in zip(t, dObj):
            self.assertTrue(_t == np.datetime64(_d))

    def test_nptimedelta64_to_dataObject(self):
        # timebase us
        t = np.array([np.timedelta64(1, "Y"), np.timedelta64(-5, "Y")])
        with self.assertRaises(RuntimeError):
            dObj = dataObject(t)

        t = np.array([np.timedelta64(1, "M"), np.timedelta64(-5, "M")])
        with self.assertRaises(RuntimeError):
            dObj = dataObject(t)

        t = np.array([np.timedelta64(1, "D"), np.timedelta64(-5, "D")])
        dObj = dataObject(t)

        for _t, _d in zip(t, dObj):
            self.assertTrue(_t == np.timedelta64(_d))

        t = np.array([np.timedelta64(1, "W"), np.timedelta64(-5, "W")])
        dObj = dataObject(t)

        for _t, _d in zip(t, dObj):
            self.assertTrue(_t == np.timedelta64(_d))

        t = np.array([np.timedelta64(1, "h"), np.timedelta64(-5, "h")])
        dObj = dataObject(t)

        for _t, _d in zip(t, dObj):
            self.assertTrue(_t == np.timedelta64(_d))

        t = np.array([np.timedelta64(1, "us"), np.timedelta64(-5, "us")])
        dObj = dataObject(t)

        for _t, _d in zip(t, dObj):
            self.assertTrue(_t == np.timedelta64(_d))

        t = np.array([np.timedelta64(1, "s"), np.timedelta64(-5, "s")])
        dObj = dataObject(t)

        for _t, _d in zip(t, dObj):
            self.assertTrue(_t == np.timedelta64(_d))

    def test_dataObject2nparray(self):
        # timedelta
        dateNp = np.array(self.tdObj, copy=False)

        # must be a copy
        self.assertTrue(dateNp.base is None)

        for a, b in zip(dateNp.ravel(), self.tdObj):
            self.assertEqual(a, b)

        # a Python timedelta object cannot be converted into Y and M
        for timebase in ["Y", "M"]:
            with self.assertRaises(ValueError):
                dateNpTimebase = np.array(
                    self.tdObj, dtype="timedelta64[%s]" % timebase
                )

        for timebase in ["W", "D", "h", "m", "s", "ms", "us", "ns", "as", "fs", "ps"]:
            dateNpTimebase = np.array(self.tdObj, dtype="timedelta64[%s]" % timebase)
            dateNpTimebase2 = dateNp.astype("timedelta64[%s]" % timebase)
            nptesting.assert_array_equal(
                dateNpTimebase, dateNpTimebase2, err_msg="Timebase: %s" % timebase
            )

        # datetime
        dateObjNoTz = self.dateObj

        for i in range(dateObjNoTz.shape[0]):
            for j in range(dateObjNoTz.shape[1]):
                for k in range(dateObjNoTz.shape[2]):
                    dateObjNoTz[i, j, k] = dateObjNoTz[i, j, k].replace(tzinfo=None)

        dateNp = np.array(dateObjNoTz, copy=False)

        # must be a copy
        self.assertTrue(dateNp.base is None)

        for a, b in zip(dateNp.ravel(), dateObjNoTz):
            self.assertEqual(a, b)

        for timebase in [
            "Y",
            "M",
            "W",
            "D",
            "h",
            "m",
            "s",
            "ms",
            "us",
            "ns",
            "as",
            "fs",
            "ps",
        ]:
            dateNpTimebase = np.array(dateObjNoTz, dtype="datetime64[%s]" % timebase)
            dateNpTimebase2 = dateNp.astype("datetime64[%s]" % timebase)
            nptesting.assert_array_equal(
                dateNpTimebase, dateNpTimebase2, err_msg="Timebase: %s" % timebase
            )

    def test_dateTimeAssign(self):
        dobj = dataObject([1, 1], "datetime")
        d1 = datetime.today()
        d2 = np.datetime64("2005-02-25")
        d3 = np.datetime64("2005-02-25T03:30")

        dobj[0, 0] = d1
        self.assertEqual(d1, dobj[0, 0])
        dobj[0, 0] = d2
        self.assertEqual(d2, np.array(dobj)[0, 0])
        dobj[0, 0] = d3
        self.assertEqual(d3, np.array(dobj)[0, 0])

    def test_timeDeltaAssign(self):
        dobj = dataObject([1, 1], "timedelta")
        d1 = timedelta(days=1, seconds=300)
        d2 = np.timedelta64(-10, "h")
        d3 = np.timedelta64(20, "D")

        dobj[0, 0] = d1
        self.assertEqual(d1, dobj[0, 0])
        dobj[0, 0] = d2
        self.assertEqual(d2, np.array(dobj)[0, 0])
        dobj[0, 0] = d3
        self.assertEqual(d3, np.array(dobj)[0, 0])

    def test_createDatetimeDataObject(self):
        a = datetime.today()
        numdays = 100
        dateList = []
        for x in range(0, numdays):
            dateList.append(a - timedelta(days=x))

        dateScale = dataObject([1, len(dateList)], "datetime", data=dateList)

        for item1, item2 in zip(dateList, dateScale):
            self.assertEqual(item1, item2)

        # with timezone
        a = datetime(2022, 3, 2, 22, 3, 45, tzinfo=timezone(timedelta(seconds=-7200)))
        dateList = []
        for x in range(0, numdays):
            dateList.append(a - timedelta(days=x))

        dateScale = dataObject([1, len(dateList)], "datetime", data=dateList)

        for item1, item2 in zip(dateList, dateScale):
            self.assertEqual(item1, item2)

        # create from one scalar
        dateScale = dataObject([2, 2], "datetime", data=a)
        for item1 in dateScale:
            self.assertEqual(item1, a)

        b1 = np.datetime64("2022-03-04")
        b2 = datetime(2022, 3, 4)
        dateScale = dataObject([2, 2], "datetime", data=b1)
        for item1 in dateScale:
            self.assertEqual(item1, b2)

    def test_createTimedeltaDataObject(self):
        numdays = 100
        dateList = []
        for x in range(0, numdays):
            dateList.append(timedelta(days=x))

        dateScale = dataObject([1, len(dateList)], "timedelta", data=dateList)

        for item1, item2 in zip(dateList, dateScale):
            self.assertEqual(item1, item2)


if __name__ == "__main__":
    unittest.main(module="dataobject_datetime", exit=False)
