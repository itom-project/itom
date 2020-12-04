import itomStubsGen as isg
import unittest
from typing import Tuple, Dict, List, Optional
import warnings

try:
    from typing import Literal  # only available from Python 3.8 on
    hasLiteral: bool = True
except ImportError:
    hasLiteral: bool = False


class ItomStubsGenTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        pass
    
    
    def test_nptypehints2typing(self):
        """Test the conversion from numpydoc-types to typing type hints."""
        
        # expected conversions
        singleTypes: Dict[str, str] = {
            "str": "str",
            "int": "int",
            "dataObject": "dataObject",
            "float": "float",
            "np.ndarray": "np.ndarray",
            "None": "None"}
        
        newValues: Dict[str, str] = {}
        
        # extend single types by :py:obj:`...` or similar refs
        for key in singleTypes:
            newValues[":py:obj:`%s`" % key] = singleTypes[key]
            newValues[":obj:`%s`" % key] = singleTypes[key]
            newValues[":py:class:`%s`" % key] = singleTypes[key]
        
        singleTypes.update(newValues)
        
        nestedTypes: Dict[str, str] = {}
        
        for key in singleTypes:
            for nestItem in ["List", "Tuple", "Sequence"]:
                s = nestItem.lower()
                val = singleTypes[key]
                nestedTypes["%s of %s" % (s, key)] = "%s[%s]" % (nestItem, val)
                nestedTypes[":obj:`%s` of %s" % (s, key)] = "%s[%s]" % (nestItem, val)
                nestedTypes[":py:obj:`%s` of %s" % (s, key)] = \
                    "%s[%s]" % (nestItem, val)
                nestedTypes[":py:class:`%s` of %s" % (s, key)] = \
                    "%s[%s]" % (nestItem, val)
        
        conversions = singleTypes
        conversions.update(nestedTypes)
        
        conversions["list of int or sequence of :class:`itom.dataObject`"] = \
            "Union[List[int], Sequence[itom.dataObject]]"
        conversions["int or None"] = "Optional[int]"
        conversions["float or None or int"] = "Optional[Union[float, int]]"
        
        if hasLiteral:
            conversions["{4}"] = "Literal[4]"
            conversions["{'test', 'fox'}"] = "Literal['test', 'fox']"
            conversions["{2, 3, 4} or int"] ="Union[Literal[2, 3, 4], int]"
        else:
            conversions["{4}"] = "Any"
            conversions["{'test', 'fox'}"] = "Any"
            conversions["{2, 3, 4} or int"] ="Union[Any, int]"
        
        for key in conversions:
            self.assertEqual(isg._nptype2typing(key), conversions[key])
    
    
    def test_parse_numpydoc_section(self):
        """Test the parser for the parameters, returns or yields section."""
        docstring1: str = \
"""Returns device information for each spectrometer that is connected.

Parameters
----------
path : :class:`str`
    The path to the Avantes SDK.
port_id : :class:`int`
    ID of port to be used. One of:

        * -1: Use both Ethernet (AS7010) and USB ports
        * 0: Use USB port
        * 1..255: Not supported in v9.7 of the SDK
        * 256: Use Ethernet port (AS7010)
nmax : :class:`int`, optional
    The maximum number of devices that can be in the list.

Returns
-------
:class:`list` of :class:`.AvsIdentityType`
    The information about the devices.
"""
        
        docstring2: str = \
"""Exceptions are documented in the same way as classes.

The __init__ method may be documented in either the class level
docstring, or as a docstring on the __init__ method itself.

Either form is acceptable, but the two should not be mixed. Choose one
convention to document the __init__ method and be consistent with it.

Note
----
Do not include the `self` parameter in the ``Parameters`` section.

Parameters
----------
msg : str
    Human readable string describing the exception.
code : :obj:`int`, optional
    Numeric error code.

Attributes
----------
msg : str
    Human readable string describing the exception.
code : int
    Numeric error code.

Returns
-------
bool
    True if successful, False otherwise.

Yields
-------
bytes
    True if successful, False otherwise.
"""
        
        docstring3: str = \
"""This is an example of a module level function.

Function parameters should be documented in the ``Parameters`` section.
The name of each parameter is required. The type and description of each
parameter is optional, but should be included if not obvious.

If ``*args`` or ``**kwargs`` are accepted,
they should be listed as ``*args`` and ``**kwargs``.

The format for a parameter is::

    name : type
        description

        The description may span multiple lines. Following lines
        should be indented to match the first line of the description.
        The ": type" is optional.

        Multiple paragraphs are supported in parameter
        descriptions.

Parameters
----------
param1 : int
    The first parameter.
param2 : :obj:`str`, optional
    The second parameter.
*args
    Variable length argument list.
**kwargs
    Arbitrary keyword arguments.

Returns
-------
bool
    True if successful, False otherwise.

    The return type is not optional. The ``Returns`` section may span
    multiple lines and paragraphs. Following lines should be indented to
    match the first line of the description.

    The ``Returns`` section supports any reStructuredText formatting,
    including literal blocks::

        {
            'param1': param1,
            'param2': param2
        }

Raises
------
AttributeError
    The ``Raises`` section is a list of all exceptions
    that are relevant to the interface.
ValueError
    If `param2` is equal to `param1`.

"""
        
        docstring4: str = \
        """This is an example of a module level function.

Parameters
----------
param1 : sequence of int or iterable of int, optional
    The first parameter.
"""
        
        docstring5: str = \
        """This is an errorneous docstring.

Parameters
----------
param1 : sequence of int, iterable of int, optional
    Union must not be comma-separated.
"""
        
        # no Yields section in docstring1
        res = isg._parse_npdoc_argsection(docstring1, "Yields")
        self.assertIsNone(res)
        
        # Yields section in docstring2
        res = isg._parse_npdoc_argsection(docstring2, "Yields")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].dtype, 'bytes')
        self.assertEqual(res[0].name, '')
        
        # no Yields section in docstring3
        res = isg._parse_npdoc_argsection(docstring3, "Yields")
        self.assertIsNone(res)
        
        self.assertEqual(isg._get_rettype_from_npdocstring(docstring1),
                         "List[.AvsIdentityType]")
        
        self.assertEqual(isg._get_rettype_from_npdocstring(docstring2),
                         "bool")
        
        self.assertEqual(isg._get_rettype_from_npdocstring(docstring3),
                         "bool")
        
        res = isg._get_parameters_from_npdocstring(docstring3)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].dtype, 'int')
        self.assertEqual(res[0].optional, False)
        self.assertEqual(res[1].dtype, 'str')
        self.assertEqual(res[1].optional, True)
        self.assertIsNone(res[2].dtype)
        self.assertEqual(res[2].name, "*args")
        self.assertEqual(res[2].optional, True)
        self.assertIsNone(res[3].dtype)
        self.assertEqual(res[3].name, "**kwargs")
        self.assertEqual(res[3].optional, True)
        
        res = isg._get_parameters_from_npdocstring(docstring4)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].dtype, "Union[Sequence[int], Iterable[int]]")
        self.assertEqual(res[0].optional, True)
        
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            res = isg._get_parameters_from_npdocstring(docstring5)
            # Verify some things
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, RuntimeWarning))
    
    def test_property_parser(self):
        
        class Test:
            @property
            def demo(self):
                """list of int or None : Returns the bounding rectangle of.
                
                The bounding rectangle is given by a list (x, y, width, height)."""
                pass
        
        text = isg._parse_property_docstring(Test.demo, 4)
        
        text2 = \
"""    @property
    def demo(self) -> Optional[List[int]]:
        \"\"\"
        Returns the bounding rectangle of.
        
        The bounding rectangle is given by a list (x, y, width, height).
        \"\"\"
        pass"""
        
        text_ = text.split("\n")
        text2_ = text2.split("\n")
        self.assertEqual(len(text_), len(text2_))
        for t, t2 in zip(text_, text2_):
            self.assertEqual(t.strip(), t2.strip())
    
    def test_get_direct_members(self):
        
        # some demo classes
        class A:
            def __init__(self):
                pass
            
            def meth1(self):
                pass
            
            def meth2(self):
                pass
            
            @staticmethod
            def meth3(self):
                pass
        
        class B(A):
            def __init_(self):
                pass
            
            def meth1(self):
                pass
            
            def meth4(self):
                pass
        
        membersA = [m for m in isg._get_direct_members(A)]
        membersA_name = [m[0] for m in membersA]
        
        for m in membersA:
            self.assertIs(type(m[0]), str)
            self.assertEqual(len(m), 3)
        
        self.assertIn("meth1", membersA_name)
        self.assertIn("meth2", membersA_name)
        self.assertIn("meth3", membersA_name)
        
        membersB = [m for m in isg._get_direct_members(B)]
        membersB_name = [m[0] for m in membersB]
        
        for m in membersA:
            self.assertIs(type(m[0]), str)
            self.assertEqual(len(m), 3)
        
        self.assertIn("meth1", membersB_name)
        self.assertIn("meth4", membersB_name)
        self.assertNotIn("meth2", membersB_name)
        self.assertNotIn("meth3", membersB_name)
    
    def test_parse_npdoc_argsection(self):
        demo1 = """lorem ipsum

lorem ipsum
lorem ipsum

Returns
lorem ipsum"""
        args = isg._parse_npdoc_argsection(demo1, "Yields")
        self.assertIsNone(args)
        
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            args = isg._parse_npdoc_argsection(demo1, "Returns")
            # Verify some things
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, RuntimeWarning))
            self.assertIsNone(args)
        
        demo2 = """Lorem ipsum

Lorem ipsum
Lorem ipsum

Parameters
----------
arg1 : int, optional
    text of arg1
arg2
    text of arg2
arg3 : int
    text of arg3
*args : list of float or tuple of int
    text of args

Returns
-------
a : float
    text
int
    value without name
"""
        for args in [
                isg._parse_npdoc_argsection(demo2, "Parameters"),
                isg._get_parameters_from_npdocstring(demo2)]:
            self.assertEqual(len(args), 4)
            
            # arg 1
            self.assertEqual(args[0].name, "arg1")
            self.assertEqual(args[0].dtype, "int")
            self.assertTrue(args[0].optional)
            
            # arg 2
            self.assertEqual(args[1].name, "arg2")
            self.assertIsNone(args[1].dtype)
            self.assertFalse(args[1].optional)
            
            # arg 3
            self.assertEqual(args[2].name, "arg3")
            self.assertEqual(args[2].dtype, "int")
            self.assertFalse(args[2].optional)
            
            # arg 4
            self.assertEqual(args[3].name, "*args")
            self.assertEqual(args[3].dtype, "Union[List[float], Tuple[int]]")
            self.assertTrue(args[3].optional)
        
        args = isg._parse_npdoc_argsection(demo2, "Returns")
        
        self.assertEqual(len(args), 2)
        
        # arg 1
        self.assertEqual(args[0].name, "a")
        self.assertEqual(args[0].dtype, "float")
        self.assertFalse(args[0].optional)
        
        # arg 2
        self.assertEqual(args[1].name, "")
        self.assertEqual(args[1].dtype, "int")
        self.assertFalse(args[1].optional)
    
    def test_parse_signature_from_first_line(self):
        
        class Demo:
            def meth1(self):
                """this method does nothing"""
                pass
            
            def meth2(self):
                """"""
                pass
            
            @staticmethod
            def meth3():
                """def meth3(arg1 -> no"""
            
            def meth4(self):
                """meth4(a, b, c) -> Optional[int]
                
                Parameters
                ----------
                a : int
                    text
                b : float, optional
                    value
                c
                    no type
                """
            
            def meth5(self):
                """myFunc() -> int"""
        
        wrong_signatures = [Demo.meth1, Demo.meth2, Demo.meth3]
        
        for ws in wrong_signatures:
            line1 = ws.__doc__.split("\n")[0]
            with self.assertRaises(ValueError, msg=ws.__qualname__):
                isg._parse_signature_from_first_line(ws, line1)
        
        meth4 = Demo.meth4
        sig_meth4 = meth4.__doc__.split("\n")[0]
        sig = isg._parse_signature_from_first_line(meth4, sig_meth4)
        self.assertEqual(sig.name, "meth4")
        self.assertEqual(sig.rettype, "Optional[int]")
        self.assertEqual(len(sig.args), 3)
        
        meth5 = Demo.meth5
        sig_meth5 = meth5.__doc__.split("\n")[0]
        
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            sig = isg._parse_signature_from_first_line(meth5, sig_meth5)
            # Verify some things
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, RuntimeWarning))
            self.assertEqual(sig.name, "meth5")
            self.assertEqual(sig.rettype, "int")
        
    def test_parse_property_docstring(self):
        
        class Demo:
            @property
            def prop1(self):
                pass
            
            @property
            def prop2(self):
                """int : text of property."""
            
            @property
            def prop3(self):
                """text of property."""
        
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            text1 = isg._parse_property_docstring(Demo.prop1, 0)
            # Verify some things
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, RuntimeWarning))
            self.assertEqual(text1, "@property\ndef prop1(self):\n    pass")
        
        text2 = isg._parse_property_docstring(Demo.prop2, 4)
        self.assertEqual(text2, "    @property\n    def prop2(self) -> int:\n        \"\"\"text of property.\"\"\"\n        pass")
        
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            text3 = isg._parse_property_docstring(Demo.prop3, 0)
            # Verify some things
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, RuntimeWarning))
            self.assertEqual(text3, "@property\ndef prop3(self):\n    \"\"\"text of property.\"\"\"\n    pass")
    
    def test_parse_args_string(self):
        args = isg._parse_args_string("a, b = None, c = Optional[Union[int, float]]")
        
        self.assertEqual(len(args), 3)
        
        # arg 1
        self.assertEqual(args[0].name, "a")
        self.assertFalse(args[0].optional)
        self.assertIsNone(args[0].dtype)
        self.assertIsNone(args[0].default)
        
        # arg 2
        self.assertEqual(args[1].name, "b")
        self.assertTrue(args[1].optional)
        self.assertIsNone(args[1].dtype)
        self.assertEqual(args[1].default, "None")
        
        # arg 3
        self.assertEqual(args[2].name, "c")
        self.assertTrue(args[2].optional)
        self.assertIsNone(args[2].dtype)
        self.assertEqual(args[2].default, "Optional[Union[int, float]]")
    
    
if __name__ == '__main__':
    unittest.main(module='itom_stubs_generator', exit=False)