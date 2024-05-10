import unittest
from itom import loadIDC, saveIDC, dataObject
import numpy as np
import tempfile
import os
from numpy import testing as nptesting


class IdcTest(unittest.TestCase):
    def test_saveLoadDefault(self):
        data = {"a": 1, "b": "string", "c": b"bytes"}

        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = os.path.join(tmpdirname, "test.idc")
            saveIDC(filename, data, overwriteIfExists=True)
            data2 = loadIDC(filename)

        self.assertEqual(data, data2)

    def test_saveLoadDataObjects(self):
        for dtype in [
            "uint8",
            "int16",
            "int32",
            "float32",
            "float64",
            "complex64",
            "datetime",
            "timedelta",
            "rgba32",
        ]:
            data = {"dobj": dataObject.zeros([5, 4, 3], dtype=dtype)}

            with tempfile.TemporaryDirectory() as tmpdirname:
                filename = os.path.join(tmpdirname, "test.idc")
                saveIDC(filename, data, overwriteIfExists=True)
                data2 = loadIDC(filename)
                arr1 = np.array(data["dobj"])
                arr2 = np.array(data2["dobj"])
                self.assertTrue(np.all(arr1 == arr2))
                self.assertTrue(arr1.dtype == arr2.dtype)

    def test_saveLoadSpecialNames(self):
        data = {
            "a": 1,
            "b": "string",
            "c": b"bytes",
            "0test": "key no identifier",
            123: "key no identifier",
        }

        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = os.path.join(tmpdirname, "test.idc")
            saveIDC(filename, data, overwriteIfExists=True)
            data2 = loadIDC(filename)

        self.assertEqual(data, data2)


if __name__ == "__main__":
    unittest.main(module="idc_test", exit=False)
