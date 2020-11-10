import itomStubsGen as isg
import unittest
from typing import Tuple, Dict, List, Optional


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

if __name__ == '__main__':
    unittest.main(module='itom_stubs_generator', exit=False)