import itomJediLib as jedilib
import unittest
from typing import Tuple, Dict, List, Optional
import warnings


class ItomJediLibTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_completions_errors(self):
        """tests errors in completions."""
        text = "range(0, 4)\nbytes()"
        
        wrong_line_cols = [(0, 12), (1, -1), (1, 20), (2, 0), (2, 1)]
        
        for line, col in wrong_line_cols:
            with self.assertRaises(ValueError):
                result = jedilib.completions(text, line, col, "", "")
    
    def test_completions_itommod_class(self):
        """completion test for classes in the itom module."""
        text = \
"""from itom import dataObject
dataObject"""
        result = jedilib.completions(text, 1, 5, path="", prefix="")
        self.assertEqual(len(result), 1)  # 1 result
        item = result[0]
        self.assertEqual(item[0], "dataObject")
        self.assertEqual(item[1], "class dataObject")
        self.assertEqual(item[2], ":/classNavigator/icons/class.png")
        
        text = \
"""from itom import dataObject
myObj = dataObject([2, 2])
myObj.base
myObj.axisScales
myObj.abs()
dataObject.ones([2, 2])"""
        
        # check that sufficient completions are loaded
        result = jedilib.completions(text, 2, 6, path="", prefix="")
        self.assertGreater(len(result), 20)
        
        # check properties
        result = jedilib.completions(text, 2, 9, path="", prefix="")
        self.assertEqual(len(result), 1)  # 1 result
        item = result[0]
        self.assertEqual(item[0], "base")
        self.assertEqual(item[1], "def base")
        self.assertEqual(item[2], ":/classNavigator/icons/var.png")
        
        # check method
        result = jedilib.completions(text, 4, 8, path="", prefix="")
        self.assertEqual(len(result), 1)  # 1 result
        item = result[0]
        self.assertEqual(item[0], "abs")
        self.assertEqual(item[1], "def abs")
        self.assertEqual(item[2], ":/classNavigator/icons/method.png")
        
        # check staic method
        result = jedilib.completions(text, 5, 13, path="", prefix="")
        self.assertEqual(len(result), 1)  # 1 result
        item = result[0]
        self.assertEqual(item[0], "ones")
        self.assertEqual(item[1], "def ones")
        self.assertEqual(item[2], ":/classNavigator/icons/method.png")
    
    def test_completions_builtins_class(self):
        """completion test for builtin classes."""
        text = "range"
        result = jedilib.completions(text, 0, 4, path="", prefix="")
        self.assertEqual(len(result), 1)  # 1 result
        item = result[0]
        self.assertEqual(item[0], "range")
        self.assertEqual(item[1], "class range")
        self.assertEqual(item[2], ":/classNavigator/icons/class.png")
        
        text = "bytes"
        result = jedilib.completions(text, 0, 5, path="", prefix="")
        self.assertEqual(len(result), 2)  # 2 results, bytes and BytesWarning
        item = result[0]
        self.assertEqual(item[0], "bytes")
        self.assertEqual(item[1], "class bytes")
        self.assertEqual(item[2], ":/classNavigator/icons/class.png")
    

if __name__ == '__main__':
    unittest.main(module='itom_jedilib', exit=False)