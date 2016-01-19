import unittest
import dataobject_squeeze_reshape
import dataobject_static_constructors

def suite():
    suite = unittest.TestSuite()
    suite.addTest (unittest.makeSuite(dataobject_squeeze_reshape.DataObjectResize))
    suite.addTest (unittest.makeSuite(dataobject_static_constructors.DataObjectStaticConstructors))
    return suite
    
if (__name__ == "__main__"):
    runner = unittest.TextTestRunner()
    test_suite = suite()
    runner.run (test_suite)