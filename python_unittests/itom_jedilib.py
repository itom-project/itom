import itomJediLib as jedilib
import unittest
from typing import Tuple, Dict, List, Optional
import warnings


class ItomJediLibTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Loads the demo scripts itom_jedilib_demo_notypehints.py."""
        with open("itom_jedilib_demo_notypehints.py", "rt") as fp:
            cls.sample_notypehints = fp.read()
    
    def _assertStartsWith(self, statement, string):
        self.assertTrue(statement.startswith(string))
    
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
    
    def test_help_notypehints(self):
        """."""
        doc = self.sample_notypehints
        p = "temp.py"
        
        # multiline comment -> no help
        h = jedilib.get_help(doc, 0, 15, path=p)
        self.assertEqual(len(h), 0)
        
        # empty line -> no help
        h = jedilib.get_help(doc, 1, 0, path=p)
        self.assertEqual(len(h), 0)
        
        # import itom
        h = jedilib.get_help(doc, 2, 3, path=p)  # import is nothing
        self.assertEqual(len(h), 0)
        h = jedilib.get_help(doc, 2, 9, path=p)  # itom is a module
        self.assertEqual(h[0][0], "module itom")
        self.assertTrue("Module itom" in h[0][1][0])
        
        # __version__ string
        h1 = jedilib.get_help(doc, 5, 3, path=p)  # __version__ variable -> help
        self.assertEqual(h1, [('__version__ = "2.0.0"', ['__version__: str'])])
        h2 = jedilib.get_help(doc, 5, 14, path=p)  # equal sign (keyword) -> no help
        self.assertEqual(len(h2), 0)
        h3 = jedilib.get_help(doc, 5, 18, path=p)  # version string -> no help
        self.assertEqual(len(h3), 0)
        
        # meth1_nodocstr
        h = jedilib.get_help(doc, 8, 1, path=p)  # keyword -> no help
        self.assertEqual(len(h), 0)
        h = jedilib.get_help(doc, 8, 12, path=p)  # method itself -> help
        self.assertEqual(h, [('def meth1_nodocstr', ['meth1_nodocstr()'])])
        
        # meth1_nodocstr, content
        h = jedilib.get_help(doc, 9, 15, path=p)  # int number
        self.assertEqual(h[0][0], 'instance int')
        self.assertTrue("int([x]) -> integer\n" in h[0][1][0])
        h = jedilib.get_help(doc, 9, 13, path=p)  # keyword + -> no help
        self.assertEqual(len(h), 0)
        h = jedilib.get_help(doc, 9, 6, path=p)  # var, no typehints, rettype None
        self.assertEqual(h, [('var = 2 + 3', ['var: None'])])
        h = jedilib.get_help(doc, 10, 6, path=p)  # var, no typehints, one expr: type: int
        self.assertEqual(h, [('var2 = 3', ['var2: int'])])
        
        # meth1_docstr
        h = jedilib.get_help(doc, 14, 1, path=p)  # keyword -> no help
        self.assertEqual(len(h), 0)
        h = jedilib.get_help(doc, 14, 12, path=p)  # method itself -> help
        self.assertEqual(
            h, 
            [('def meth1_docstr',
              ['meth1_docstr()\n\nReturns the float result of 2 + 3.'])
            ]
        )
        
        # meth2_nodocstr
        h = jedilib.get_help(doc, 20, 14, path=p)  # signature
        self.assertEqual(h, [('def meth2_nodocstr', ['meth2_nodocstr(arg1, arg2=4.0)'])])
        
        h = jedilib.get_help(doc, 20, 22, path=p)  # arg1
        h2 = jedilib.get_help(doc, 21, 11, path=p)  # the same arg1
        self.assertEqual(h, [('param arg1', ['param arg1'])])
        self.assertEqual(h, h2)
        
        h = jedilib.get_help(doc, 20, 28, path=p)  # arg2
        h2 = jedilib.get_help(doc, 21, 18, path=p)  # the same arg2
        self.assertEqual(h, [('param arg2=4.0', ['param arg2=4.0'])])
        self.assertEqual(h, h2)
        
        h = jedilib.get_help(doc, 26, 14, path=p)  # dobj3.shape, word 'dobj3'
        self.assertEqual(h, [('dobj3 = dobj.reshape([3, 2])', ['dobj3: dataObject'])])
        h = jedilib.get_help(doc, 26, 19, path=p)  # dobj3.shape, word 'shape'
        self.assertEqual(len(h), 1)
        self.assertEqual(h[0][0], "def shape")
        self._assertStartsWith(h[0][1][0], "shape: tuple\n\nGets the shape")
        

if __name__ == '__main__':
    unittest.main(module='itom_jedilib', exit=False)