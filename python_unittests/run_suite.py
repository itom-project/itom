import unittest
import dataobject_squeeze_reshape
import dataobject_static_constructors
import dataobject_mapping
import dataobject_np_conversion
import dataobject_makecontinuous
import plot_test

def suite():
    suite = unittest.TestSuite()
    suite.addTest (unittest.makeSuite(dataobject_squeeze_reshape.DataObjectResize))
    suite.addTest (unittest.makeSuite(dataobject_static_constructors.DataObjectStaticConstructors))
    suite.addTest (unittest.makeSuite(dataobject_mapping.DataObjectMapping))
    suite.addTest (unittest.makeSuite(dataobject_np_conversion.DataObjectNpConversion))
    suite.addTest (unittest.makeSuite(dataobject_makecontinuous.DataObjectMakeContinuous))
    suite.addTest (unittest.makeSuite(plot_test.PlotTest))
    return suite
    
if (__name__ == "__main__"):
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = suite()
    runner.run (test_suite)