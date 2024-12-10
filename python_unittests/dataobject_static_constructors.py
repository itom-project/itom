import unittest
from itom import dataObject


class DataObjectStaticConstructors(unittest.TestCase):
    """
    Unit tests for the DataObject static constructors.
    This module contains unit tests for verifying the behavior of static constructors
    of the `dataObject` class. The tests ensure that the static constructors return
    objects with the expected data types and shapes.
    Classes:
        DataObjectStaticConstructors: Contains unit tests for `dataObject` static constructors.
    Methods:
        setUpClass: Sets up the test class (currently does nothing).
        test_default_types: Tests that all static constructors return `uint8` data type by default.
        test_stack_dimensions: Tests the dimensions of stacked `dataObject` instances.
    """
    @classmethod
    def setUpClass(cls):
        pass

    def test_default_types(self):
        """all static constructors should return uint8 if not otherwise
        indicated (like the default constructor)"""
        obj = dataObject.zeros([10, 10])
        self.assertEqual(obj.dtype, "uint8")

        obj = dataObject.ones([10, 10])
        self.assertEqual(obj.dtype, "uint8")

        obj = dataObject.randN([10, 10])
        self.assertEqual(obj.dtype, "uint8")

        obj = dataObject.rand([10, 10])
        self.assertEqual(obj.dtype, "uint8")

        obj = dataObject.eye(5)
        self.assertEqual(obj.dtype, "uint8")

    def test_stack_dimensions(self):
        obj1 = dataObject([2, 2])
        obj2 = dataObject([2, 2])
        self.assertEqual(dataObject.dstack([obj1, obj2]).shape, (2, 2, 2))

        obj1 = dataObject([1, 3, 1, 1, 2, 2])
        obj2 = dataObject([5, 1, 2, 2])
        self.assertEqual(dataObject.dstack([obj1, obj2]).shape, (8, 2, 2))

        obj1 = dataObject([2, 3])
        obj2 = dataObject([2, 2])
        self.assertRaises(TypeError, dataObject.dstack, (obj1, obj2))


if __name__ == "__main__":
    unittest.main(module="dataobject_static_constructors", exit=False)
