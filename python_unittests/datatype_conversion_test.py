import unittest
from itom import ui
import numpy as np
from numpy import testing as nptesting


class DatatypeConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gui = ui("datatype_conversion.ui")
        cls.widget = cls.gui.menuComboBox

    @classmethod
    def tearDownClass(cls):
        del cls.gui

    def test_qmetatype_number_conversion(self):
        cls = DatatypeConversionTest

        result = cls.widget.call("__unittestVariantList", [])
        self.assertEqual(result, tuple())

        numbers = (
            1.5,
            1,
            np.int8(1),
            np.uint8(2),
            np.int16(234),
            np.uint16(4000),
            np.int32(-64345345),
            np.int32(234234234),
            np.int64(-(2 ** 50)),
            np.uint64(2 ** 64 - 1),
        )
        result = cls.widget.call("__unittestVariantList", numbers)
        self.assertEqual(
            result,
            (1.5, 1, 1, 2, 234, 4000, -64345345, 234234234, -(2 ** 50), 2 ** 64 - 1),
        )

        # check int type
        numbers = (
            1,
            np.int8(1),
            np.uint8(2),
            np.int16(234),
            np.uint16(4000),
            np.int32(-64345345),
            # np.uint32(234234234),
        )
        for num in numbers:
            result = cls.widget.call("__unittestInt", num)
            self.assertEqual(result, num)

        # check float type
        float_numbers = (-7.55, np.float32(23.78), np.float64(2113.123545345345234))
        for num in float_numbers + numbers:
            result = cls.widget.call("__unittestFloat", num)
            nptesting.assert_approx_equal(result, num)
            result = cls.widget.call("__unittestDouble", num)
            nptesting.assert_approx_equal(result, num)

        # check short type
        numbersShort = (
            1,
            np.int8(1),
            np.uint8(2),
            np.int16(234),
            np.uint16(4000),
            np.int32(-32768),
            np.uint32(32767),
        )
        for num in numbersShort:
            result = cls.widget.call("__unittestShort", num)
            self.assertEqual(result, num)

        # check int64 type
        numbersInt64 = (
            1,
            np.int8(1),
            np.uint8(2),
            np.int16(234),
            np.uint16(4000),
            np.int32(-32768),
            np.uint32(32767),
            np.int64(-(2 ** 50)),
        )
        for num in numbersInt64:
            result = cls.widget.call("__unittestInt64", num)
            self.assertEqual(result, num)

        # check int64 type
        numbersUInt64 = (
            1,
            np.int8(1),
            np.uint8(2),
            np.int16(234),
            np.uint16(4000),
            np.int32(32768),
            np.uint32(32767),
            np.int64(2 ** 50),
            np.uint64(2 ** 64 - 1),
        )
        for num in numbersUInt64:
            result = cls.widget.call("__unittestUInt64", num)
            self.assertEqual(result, num)


if __name__ == "__main__":
    unittest.main(module="datatype_conversion_test", exit=False, verbosity=10)
