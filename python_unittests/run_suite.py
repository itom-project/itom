import sys
import unittest

import dataobject_comparison
import dataobject_constructors
import dataobject_datetime
import dataobject_makecontinuous
import dataobject_mapping
import dataobject_multiplication
import dataobject_np_conversion
import dataobject_scale_offset
import dataobject_squeeze_reshape
import dataobject_static_constructors
import datatype_conversion_test
import idc_test
import plot_test
import pointcloud_pickle
import shape_test

pyversion = (sys.version_info.major, sys.version_info.minor)
if pyversion >= (3, 6):
    import itom_algorithm_stubs_generator
    import itom_jedilib
    import itom_stubs_generator


def suite():
    """
    This module defines a test suite for running unit tests on various components.
    The suite function aggregates multiple test cases from different modules into a single test suite.
    It includes tests for data object operations, plotting, shape tests, datatype conversions, point cloud pickling,
    and IDC tests. Additionally, if the Python version is 3.6 or higher, it includes tests for Itom stubs generation,
    Itom Jedi library, and Itom algorithm stubs generation.
    Functions:
        suite(): Creates and returns a unittest.TestSuite object containing all the specified test cases.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_squeeze_reshape.DataObjectResize
        )
    )
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_static_constructors.DataObjectStaticConstructors
        )
    )
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_constructors.DataObjectConstructors
        )
    )
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_comparison.DataObjectComparison
        )
    )
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_mapping.DataObjectMapping
        )
    )
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_multiplication.DataObjectMultiplication
        )
    )
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_np_conversion.DataObjectNpConversion
        )
    )
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_makecontinuous.DataObjectMakeContinuous
        )
    )
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_scale_offset.DataObjectScaleOffset
        )
    )
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_datetime.DataObjectDatetime
        )
    )
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(plot_test.PlotTest))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(shape_test.ShapeTest)
    )
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            datatype_conversion_test.DatatypeConversionTest
        )
    )
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            pointcloud_pickle.PointCloudPickle
        )
    )
    test_suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(idc_test.IdcTest))

    if pyversion >= (3, 6):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(
                itom_stubs_generator.ItomStubsGenTest
            )
        )
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(
                itom_jedilib.ItomJediLibTest
            )
        )
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(
                itom_algorithm_stubs_generator.ItomAlgorithmsStubsGenTest
            )
        )
    return test_suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    testSuite = suite()
    runner.run(testSuite)
