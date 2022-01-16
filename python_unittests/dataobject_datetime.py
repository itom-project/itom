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
        self.dt1 = datetime(2000, 10, 10, 4, 30, 10, 6754, tzinfo=tz)
        self.dt2 = datetime(2022, 1, 1, 0, 0, 0, 0)
        self.dateObj[0, :, :] = self.dt1
        self.dateObj[1, :, :] = self.dt2

        self.tdObj = dataObject([2, 4, 3], "timedelta")
        self.td1 = timedelta(-1, 5, 0, 0, 30, 6)
        self.td2 = timedelta(1, 1, 1, 1, 1, 1)
        self.tdObj[0, :, :] = self.td1
        self.tdObj[1, :, :] = self.td2

    def test_operator_add_dataObjects(self):
        # test 1. it is not allowed to add any datatype to a datetime
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

        ## OUT-OF-PLACE ADD
        
        # test 2: adding a datetime and timedelta is allowed
        result = self.dateObj + self.tdObj

        self.assertEqual(result.dtype, "datetime")

        for val in result[0,:,:]:
            self.assertAlmostEqual(val, self.dt1 + self.td1)

        for val in result[1,:,:]:
            self.assertAlmostEqual(val, self.dt2 + self.td2)

        # test 3: adding a timedelta and datetime is allowed
        result = self.tdObj + self.dateObj
        
        self.assertEqual(result.dtype, "datetime")
        
        for val in result[0,:,:]:
            self.assertAlmostEqual(val, self.dt1 + self.td1)
        
        for val in result[1,:,:]:
            self.assertAlmostEqual(val, self.dt2 + self.td2)

        # test 4: adding a timedelta and timedelta is allowed
        result = self.tdObj + self.tdObj

        self.assertEqual(result.dtype, "timedelta")
        
        for val in result[0,:,:]:
            self.assertAlmostEqual(val, self.td1 + self.td1)
        
        for val in result[1,:,:]:
            self.assertAlmostEqual(val, self.td2 + self.td2)

        ## IN-PLACE ADD
        
        # test 2: adding a timedelta to adatetime is allowed
        result = self.dateObj.copy()
        result += self.tdObj
        
        self.assertEqual(result.dtype, "datetime")
        
        for val in result[0,:,:]:
            self.assertAlmostEqual(val, self.dt1 + self.td1)
        
        for val in result[1,:,:]:
            self.assertAlmostEqual(val, self.dt2 + self.td2)
        
        # test 3: adding a datetime to a timedelta is not allowed
        result = self.tdObj.copy()
        
        with self.assertRaises(TypeError):
            result += self.dateObj
        
        # test 4: adding a timedelta to a timedelta is allowed
        result = self.tdObj.copy()
        result += self.tdObj
        
        self.assertEqual(result.dtype, "timedelta")
        
        for val in result[0,:,:]:
            self.assertAlmostEqual(val, self.td1 + self.td1)
        
        for val in result[1,:,:]:
            self.assertAlmostEqual(val, self.td2 + self.td2)


if __name__ == "__main__":
    unittest.main(module="dataobject_datetime", exit=False)
