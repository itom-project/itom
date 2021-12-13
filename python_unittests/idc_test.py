import unittest
from itom import loadIDC, saveIDC, dataObject
import numpy as np
import tempfile
import os

class IdcTest(unittest.TestCase):

    def test_saveLoadDefault(self):
        data = {"a":1, "b":"string", "c":b"bytes"}

        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = os.path.join(tmpdirname, "test.idc")
            saveIDC(filename, data, overwriteIfExists=True)
            data2 = loadIDC(filename)

        self.assertEqual(data, data2)
        
    def test_saveLoadSpecialNames(self):
        data = {"a":1, "b":"string", "c":b"bytes", "0test":"key no identifier", 123:"key no identifier"}
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = os.path.join(tmpdirname, "test.idc")
            saveIDC(filename, data, overwriteIfExists=True)
            data2 = loadIDC(filename)
        
        self.assertEqual(data, data2)


if __name__ == '__main__':
    unittest.main(module='idc_test', exit=False)