import math
import unittest

import numpy as np
from itom import dataObject

REAL_TYPES = ["uint8", "int8", "uint16", "int16", "int32", "float32", "float64"]
UNSUPPORTED_TYPES = ["complex64", "complex128", "rgba32", "datetime", "timedelta"]

# name of the dataObject method -> numpy function with the same meaning
FLOAT_METHODS = {
    "sum": np.sum,
    "nansum": np.nansum,
    "mean": np.mean,
    "nanmean": np.nanmean,
    "min": np.min,
    "max": np.max,
    "nanmin": np.nanmin,
    "nanmax": np.nanmax,
    "ptp": np.ptp,
    "median": np.median,
    "nanmedian": np.nanmedian,
    "var": np.var,
    "std": np.std,
    "nanvar": np.nanvar,
    "nanstd": np.nanstd,
}
INT_METHODS = {
    "argmin": np.argmin,
    "argmax": np.argmax,
    "nanargmin": np.nanargmin,
    "nanargmax": np.nanargmax,
    "count_nonzero": np.count_nonzero,
}
BOOL_METHODS = {"all": np.all, "any": np.any}
NO_ARG_METHODS = list(FLOAT_METHODS) + list(INT_METHODS) + list(BOOL_METHODS)


class DataObjectReductions(unittest.TestCase):
    """numpy-like members of dataObject: sum, mean, min, max, argmin, ..., clip."""

    @staticmethod
    def _arange(dtype, shape=(3, 4, 5)):
        """returns (dataObject, np.ndarray) with the values 0, 7, 14, ... (mod 100)."""
        num = int(np.prod(shape))
        arr = ((np.arange(num) * 7) % 100).reshape(shape).astype(dtype)
        return dataObject(arr), arr

    @staticmethod
    def _withNaN():
        arr = np.array(
            [[4.0, np.nan, -2.0, 0.0], [8.0, 1.0, np.inf, -3.5]], dtype="float64"
        )
        return dataObject(arr), arr

    def _assertSame(self, value, expected, msg):
        if isinstance(expected, float) and math.isnan(expected):
            self.assertTrue(math.isnan(value), msg)
        else:
            self.assertAlmostEqual(value, expected, places=4, msg=msg)

    def test_values_like_numpy(self):
        for dtype in REAL_TYPES:
            obj, arr = self._arange(dtype)

            for name, func in FLOAT_METHODS.items():
                result = getattr(obj, name)()
                self.assertIs(type(result), float, f"{name} {dtype}")
                # dtype float64 avoids the integer overflow of numpy for small integer types
                expected = float(func(arr.astype("float64")))
                self._assertSame(result, expected, f"{name} {dtype}")

            for name, func in INT_METHODS.items():
                result = getattr(obj, name)()
                self.assertIs(type(result), int, f"{name} {dtype}")
                self.assertEqual(result, int(func(arr)), f"{name} {dtype}")

            for name, func in BOOL_METHODS.items():
                result = getattr(obj, name)()
                self.assertIs(type(result), bool, f"{name} {dtype}")
                self.assertEqual(result, bool(func(arr)), f"{name} {dtype}")

    def test_negative_values(self):
        for dtype in ["int8", "int16", "int32", "float32", "float64"]:
            arr = np.array([[-5, 3, -128], [7, 0, 127]], dtype=dtype)
            obj = dataObject(arr)
            self.assertEqual(obj.min(), -128.0)
            self.assertEqual(obj.max(), 127.0)
            self.assertEqual(obj.ptp(), 255.0)
            self.assertEqual(obj.sum(), 4.0)
            self.assertEqual(obj.argmin(), 2)
            self.assertEqual(obj.argmax(), 5)
            self.assertEqual(obj.count_nonzero(), 5)
            self.assertFalse(obj.all())
            self.assertTrue(obj.any())

    def test_roi(self):
        # the result must only depend on the values within the ROI. The indices are flat
        # indices within the ROI.
        for dtype in REAL_TYPES:
            obj, arr = self._arange(dtype, (4, 6, 7))
            roiObj = obj[1:3, 2:5, 1:6]
            roiArr = arr[1:3, 2:5, 1:6]

            for name, func in FLOAT_METHODS.items():
                expected = float(func(roiArr.astype("float64")))
                self._assertSame(getattr(roiObj, name)(), expected, f"{name} {dtype}")

            for name, func in INT_METHODS.items():
                self.assertEqual(
                    getattr(roiObj, name)(), int(func(roiArr)), f"{name} {dtype}"
                )

            index = np.unravel_index(roiObj.argmax(), roiObj.shape)
            index = tuple(int(i) for i in index)
            self.assertEqual(roiObj[index], roiArr.max())

    def test_nan_propagates(self):
        obj, arr = self._withNaN()

        for name in ["sum", "mean", "min", "max", "ptp", "median", "var", "std"]:
            self.assertTrue(math.isnan(getattr(obj, name)()), name)

        # like numpy: index of the first NaN
        self.assertEqual(obj.argmin(), 1)
        self.assertEqual(obj.argmax(), 1)
        self.assertEqual(obj.argmin(), int(np.argmin(arr)))
        self.assertEqual(obj.argmax(), int(np.argmax(arr)))
        self.assertEqual(obj.count_nonzero(), int(np.count_nonzero(arr)))

    def test_nan_ignored(self):
        obj, arr = self._withNaN()
        self.assertEqual(obj.nanmin(), -3.5)
        self.assertEqual(obj.nanmax(), math.inf)
        self.assertEqual(obj.nansum(), math.inf)
        self.assertEqual(obj.nanargmin(), int(np.nanargmin(arr)))
        self.assertEqual(obj.nanargmax(), int(np.nanargmax(arr)))

        arr[1, 2] = 2.5  # no Inf any more
        obj = dataObject(arr)

        for name in ["nansum", "nanmean", "nanmedian", "nanvar", "nanstd"]:
            self.assertAlmostEqual(
                getattr(obj, name)(), float(FLOAT_METHODS[name](arr)), msg=name
            )

    def test_only_nan(self):
        obj = dataObject.nans([2, 3], "float32")
        self.assertEqual(obj.nansum(), 0.0)

        for name in ["nanmean", "nanmin", "nanmax", "nanmedian", "nanvar", "nanstd"]:
            self.assertTrue(math.isnan(getattr(obj, name)()), name)

        with self.assertRaises(ValueError):
            obj.nanargmin()

        with self.assertRaises(ValueError):
            obj.nanargmax()

    def test_ddof(self):
        for dtype in REAL_TYPES:
            obj, arr = self._arange(dtype)
            arr = arr.astype("float64")

            for name in ["var", "std", "nanvar", "nanstd"]:
                func = FLOAT_METHODS[name]
                method = getattr(obj, name)
                self.assertIs(type(method(ddof=1)), float)
                self.assertAlmostEqual(
                    method(ddof=1), float(func(arr, ddof=1)), places=4
                )
                self.assertAlmostEqual(
                    method(ddof=2), float(func(arr, ddof=2)), places=4
                )
                self.assertAlmostEqual(method(ddof=0), method(), places=10)

                with self.assertRaises(TypeError):
                    method(ddof="a")

            # ddof is a keyword-only argument and there is no axis argument
            for name in ["nanvar", "nanstd"]:
                with self.assertRaises(TypeError):
                    getattr(obj, name)(1)

                with self.assertRaises(TypeError):
                    getattr(obj, name)(axis=0)

    def test_no_arguments_accepted(self):
        obj, _ = self._arange("float32")
        names = ["nansum", "nanmean", "nanmin", "nanmax", "median", "nanmedian"]
        names += ["nanargmin", "nanargmax", "count_nonzero"]

        for name in names:
            with self.assertRaises(TypeError, msg=name):
                getattr(obj, name)(0)

    def test_numpy_functions_unchanged(self):
        # numpy.sum(obj) calls obj.sum(axis=None, out=None) if this method exists. The result
        # must be the same as for a numpy.ndarray (value and type).
        names = ["sum", "mean", "min", "max", "ptp", "var", "std", "argmin", "argmax"]
        names += ["all", "any", "median", "count_nonzero", "nansum", "nanmax"]

        for dtype in REAL_TYPES + ["complex64", "complex128"]:
            obj, arr = self._arange(dtype, (2, 3, 4))

            for name in names:
                if dtype.startswith("complex") and name in ["ptp", "median", "nanmax"]:
                    continue

                func = getattr(np, name)
                result = func(obj)
                expected = func(arr)
                self.assertIs(type(result), type(expected), f"{name} {dtype}")
                self.assertEqual(result, expected, f"{name} {dtype}")

            self.assertEqual(np.var(obj, ddof=1), np.var(arr, ddof=1))
            self.assertTrue(np.array_equal(np.sum(obj, axis=0), np.sum(arr, axis=0)))
            self.assertTrue(np.array_equal(np.sum(obj, 1), np.sum(arr, 1)))
            self.assertTrue(
                np.array_equal(np.argmax(obj, axis=2), np.argmax(arr, axis=2))
            )

        # numpy.clip still returns a numpy.ndarray
        obj, arr = self._arange("uint8")
        result = np.clip(obj, 10, 60)
        self.assertIs(type(result), np.ndarray)
        self.assertTrue(np.array_equal(result, np.clip(arr, 10, 60)))

    def test_arguments_are_forwarded_to_numpy(self):
        names = ["sum", "mean", "min", "max", "var", "std", "argmin", "argmax"]
        names += ["all", "any"]

        for dtype in REAL_TYPES:
            obj, arr = self._arange(dtype, (2, 3, 4))

            for name in names:
                method = getattr(obj, name)
                func = getattr(np, name)

                for result, expected in [
                    (method(axis=1), func(arr, axis=1)),
                    (method(0), func(arr, 0)),
                    (method(axis=None), func(arr, axis=None)),
                ]:
                    self.assertIs(type(result), type(expected), f"{name} {dtype}")
                    self.assertTrue(np.array_equal(result, expected), f"{name} {dtype}")

            result = obj.sum(axis=2, keepdims=True, dtype="float64")
            self.assertEqual(result.shape, (2, 3, 1))
            self.assertEqual(result.dtype, np.float64)
            self.assertTrue(
                np.array_equal(obj.var(axis=0, ddof=1), np.var(arr, axis=0, ddof=1))
            )

            with self.assertRaises(TypeError):
                obj.sum(unknownArgument=2)

            with self.assertRaises(ValueError):  # numpy.exceptions.AxisError
                obj.sum(axis=5)

        # ROI and non-continuous objects
        obj, arr = self._arange("int16", (4, 6, 7))
        self.assertTrue(
            np.array_equal(
                obj[1:3, 2:5, 1:6].sum(axis=0), arr[1:3, 2:5, 1:6].sum(axis=0)
            )
        )
        obj = dataObject([3, 4, 5], "float32", continuous=0)
        obj[:, :, :] = 2
        self.assertTrue(np.array_equal(obj.sum(axis=0), np.full((4, 5), 6.0)))

    def test_empty(self):
        obj = dataObject()
        self.assertEqual(obj.sum(), 0.0)
        self.assertEqual(obj.nansum(), 0.0)
        self.assertEqual(obj.count_nonzero(), 0)
        self.assertTrue(obj.all())
        self.assertFalse(obj.any())

        for name in ["mean", "nanmean", "median", "nanmedian", "var", "std"]:
            self.assertTrue(math.isnan(getattr(obj, name)()), name)

        # numpy raises a ValueError, too
        for name in ["min", "max", "ptp", "argmin", "argmax", "nanargmin", "nanargmax"]:
            with self.assertRaises(ValueError, msg=name):
                getattr(obj, name)()

        self.assertEqual(obj.clip(0, 1).ndim, 0)

    def test_unsupported_types(self):
        for dtype in UNSUPPORTED_TYPES:
            obj = dataObject([2, 3], dtype)

            for name in NO_ARG_METHODS:
                with self.assertRaises(TypeError, msg=f"{name} {dtype}"):
                    getattr(obj, name)()

            with self.assertRaises(TypeError, msg=f"clip {dtype}"):
                obj.clip(0, 1)

    def test_clip_like_numpy(self):
        for dtype in REAL_TYPES:
            obj, arr = self._arange(dtype)

            for a_min, a_max in [(10, 60), (None, 60), (10, None), (-1000, 1e12)]:
                result = obj.clip(a_min, a_max)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.shape, obj.shape)
                expected = np.clip(arr.astype("float64"), a_min, a_max)
                self.assertTrue(
                    np.array_equal(np.array(result), expected.astype(dtype)),
                    f"{dtype} {a_min} {a_max}",
                )

            # keyword arguments
            result = obj.clip(a_max=45)
            self.assertEqual(result.max(), min(45, arr.max()))
            self.assertEqual(result.min(), arr.min())
            result = obj.clip(a_min=20)
            self.assertEqual(result.min(), 20.0)
            self.assertEqual(result.max(), arr.max())

            # the source is not changed
            self.assertTrue(np.array_equal(np.array(obj), arr))

    def test_clip_integer_bounds_are_rounded(self):
        obj, _ = self._arange("uint8")
        result = obj.clip(2.6, 10.4)
        self.assertEqual(result.min(), 3.0)
        self.assertEqual(result.max(), 10.0)

    def test_clip_nan_and_inf(self):
        obj, _ = self._withNaN()
        result = obj.clip(0, 5)
        self.assertTrue(math.isnan(result[0, 1]))  # NaN stays NaN
        self.assertEqual(result[1, 2], 5.0)  # Inf is clipped
        self.assertEqual(result[0, 2], 0.0)
        self.assertEqual(result.nanmin(), 0.0)
        self.assertEqual(result.nanmax(), 5.0)

        result = obj.clip(0, math.inf)
        self.assertEqual(result[1, 2], math.inf)

    def test_clip_meta_information_and_roi(self):
        obj, arr = self._arange("int16", (4, 6, 7))
        obj.setTag("sensor", "abc")
        obj.axisScales = (1.0, 0.5, 0.25)
        obj.axisOffsets = (0.0, 1.0, 2.0)
        obj.axisUnits = ("", "mm", "mm")
        obj.valueUnit = "nm"

        roiObj = obj[1:3, 2:5, 1:6]
        result = roiObj.clip(20, 70)
        self.assertEqual(result.shape, roiObj.shape)
        self.assertTrue(
            np.array_equal(np.array(result), np.clip(arr[1:3, 2:5, 1:6], 20, 70))
        )
        self.assertEqual(result.tags["sensor"], "abc")
        self.assertEqual(result.axisScales, roiObj.axisScales)
        self.assertEqual(result.axisOffsets, roiObj.axisOffsets)
        self.assertEqual(result.axisUnits, roiObj.axisUnits)
        self.assertEqual(result.valueUnit, "nm")

        # deep copy: the source is not changed by a change in the result
        result[0, 0, 0] = 99
        self.assertEqual(roiObj[0, 0, 0], arr[1, 2, 1])
        self.assertIsNone(result.base)

    def test_clip_invalid_arguments(self):
        obj, _ = self._arange("float64")

        with self.assertRaises(ValueError):
            obj.clip()

        with self.assertRaises(ValueError):
            obj.clip(None, None)

        with self.assertRaises(ValueError):
            obj.clip(5, 4)

        with self.assertRaises(ValueError):
            obj.clip(math.nan, 4)

        with self.assertRaises(ValueError):
            obj.clip(0, math.nan)

        with self.assertRaises(TypeError):
            obj.clip("a", 4)

        with self.assertRaises(TypeError):
            obj.clip(0, [1, 2])

        with self.assertRaises(TypeError):
            obj.clip(0, 1, 2)

    def test_numpy_scalars_as_arguments(self):
        obj, _ = self._arange("float32")
        result = obj.clip(np.float32(10), np.int16(20))
        self.assertEqual(result.min(), 10.0)
        self.assertEqual(result.max(), 20.0)


if __name__ == "__main__":
    unittest.main(module="dataobject_reductions", exit=False)
