import unittest
from itom import dataObject
import numpy as np


class DataObjectMakeContinuous(unittest.TestCase):
    """
    Unit tests for the DataObject class's makeContinuous method.
    This module contains a test case for verifying the functionality of the
    makeContinuous method in the DataObject class. The tests ensure that the
    makeContinuous method correctly converts non-continuous data objects into
    continuous ones while preserving the data integrity.
    Classes:
        DataObjectMakeContinuous: A test case class for testing the makeContinuous
        method of the DataObject class.
    Methods:
        setUpClass: A class method to set up any state that is shared across tests.
        test_dataObjectMakeContinuous: Tests the makeContinuous method with various
        slicing operations to ensure data integrity is maintained.
    """
    @classmethod
    def setUpClass(cls):
        pass

    def test_dataObjectMakeContinuous(self):
        data = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19)
        sum = np.sum(data)
        a = dataObject([2, 3, 3], "uint8", continuous=0, data=data)
        ac = a.makeContinuous()
        for i, j in zip(a, ac):
            self.assertEqual(i, j)

        b = a[:, 1:3, :]
        bc = b.makeContinuous()
        for i, j in zip(b, bc):
            self.assertEqual(i, j)

        b = a[:, 1:3, 1:3]
        bc = b.makeContinuous()
        for i, j in zip(b, bc):
            self.assertEqual(i, j)

        b = a[1:3, 1:3, 1:3]
        bc = b.makeContinuous()
        for i, j in zip(b, bc):
            self.assertEqual(i, j)


if __name__ == "__main__":
    unittest.main(module="dataobject_makecontinuous", exit=False)
