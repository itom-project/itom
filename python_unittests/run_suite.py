import sys

pyversion = "%i.%i" % \
    (sys.version_info.major, sys.version_info.minor)


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
import plot_test
import shape_test
if pyversion >= "3.6":
    import itom_stubs_generator
    import itom_jedilib


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(dataobject_squeeze_reshape.DataObjectResize))
    suite.addTest(unittest.makeSuite(dataobject_static_constructors.DataObjectStaticConstructors))
    suite.addTest(unittest.makeSuite(dataobject_constructors.DataObjectConstructors))
    suite.addTest(unittest.makeSuite(dataobject_comparison.DataObjectComparison))
    suite.addTest(unittest.makeSuite(dataobject_mapping.DataObjectMapping))
    suite.addTest (unittest.makeSuite(dataobject_multiplication.DataObjectMultiplication))
    suite.addTest(unittest.makeSuite(dataobject_np_conversion.DataObjectNpConversion))
    suite.addTest(unittest.makeSuite(dataobject_makecontinuous.DataObjectMakeContinuous))
    suite.addTest(unittest.makeSuite(dataobject_scale_offset.DataObjectScaleOffset))
    suite.addTest(unittest.makeSuite(plot_test.PlotTest))
    suite.addTest(unittest.makeSuite(shape_test.ShapeTest))
    if pyversion >= "3.6":
        suite.addTest(unittest.makeSuite(itom_stubs_generator.ItomStubsGenTest))
        suite.addTest(unittest.makeSuite(itom_jedilib.ItomJediLibTest))
    return suite


if (__name__ == "__main__"):
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = suite()
    runner.run(test_suite)