import unittest
import dataobject_squeeze_reshape

def suite():
    suite = unittest.TestSuite()
    suite.addTest (unittest.makeSuite(dataobject_squeeze_reshape.DataObjectResize))
    return suite
    
if (__name__ == "__main__"):
    runner = unittest.TextTestRunner()
    test_suite = suite()
    runner.run (test_suite)