import sys

pyversion = (sys.version_info.major, sys.version_info.minor)


import unittest
import dataobject_squeeze_reshape
import dataobject_static_constructors
import dataobject_constructors
import dataobject_comparison
import dataobject_mapping
import dataobject_multiplication
import dataobject_np_conversion
import dataobject_makecontinuous
import dataobject_scale_offset
import dataobject_datetime
import datatype_conversion_test
import pointcloud_pickle
import idc_test
import plot_test
import shape_test

if pyversion >= (3, 6):
    import itom_stubs_generator
    import itom_algorithm_stubs_generator
    import itom_jedilib


def suite():
    suite = unittest.TestSuite()
    suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_squeeze_reshape.DataObjectResize
        )
    )
    suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_static_constructors.DataObjectStaticConstructors
        )
    )
    suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_constructors.DataObjectConstructors
        )
    )
    suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_comparison.DataObjectComparison
        )
    )
    suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_mapping.DataObjectMapping
        )
    )
    suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_multiplication.DataObjectMultiplication
        )
    )
    suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_np_conversion.DataObjectNpConversion
        )
    )
    suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_makecontinuous.DataObjectMakeContinuous
        )
    )
    suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_scale_offset.DataObjectScaleOffset
        )
    )
    suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            dataobject_datetime.DataObjectDatetime
        )
    )
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(plot_test.PlotTest))
    suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(shape_test.ShapeTest)
    )
    suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            datatype_conversion_test.DatatypeConversionTest
        )
    )
    suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            pointcloud_pickle.PointCloudPickle
        )
    )
    suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(idc_test.IdcTest))

    if pyversion >= (3, 6):
        suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(
                itom_stubs_generator.ItomStubsGenTest
            )
        )
        suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(
                itom_jedilib.ItomJediLibTest
            )
        )
        suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(
                itom_algorithm_stubs_generator.ItomAlgorithmsStubsGenTest
            )
        )
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = suite()
    runner.run(test_suite)
