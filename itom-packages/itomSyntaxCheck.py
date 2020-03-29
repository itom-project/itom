"""This module builds a bridge between the syntax and style checks
of itom and different possible packages in Python

Currently this script supports the Python packages:

* pyflakes
* flake8
"""
try:
    from flake8.api import legacy as flake8
    _HAS_FLAKE8: bool = True
except ModuleNotFoundError:
    _HAS_FLAKE8: bool = False

if _HAS_FLAKE8:
    from flake8.formatting import base
    
    if False:  # `typing.TYPE_CHECKING` was introduced in 3.5.2
        from flake8.style_guide import Violation

try:
    from pyflakes import api as pyflakesapi
    _HAS_PYFLAKES: bool = True
except ModuleNotFoundError:
    _HAS_PYFLAKES: bool = False

import tempfile
import os
from typing import List
import json

###############################################################
import itom
_PUBLIC_ITOM_MODULES = [item for item in dir(itom) if not item.startswith("__")]

_CHECKER_CACHE = {} #cache to temporarily store further information

_CONFIG_DEFAULTS = {"codeCheckerPyFlakesCategory":2,
        "codeCheckerFlake8Docstyle": "pep257",
        "codeCheckerFlake8ErrorNumbers": "F",
        "codeCheckerFlake8IgnoreEnabled": False,
        "codeCheckerFlake8IgnoreValues": "",
        "codeCheckerFlake8MaxComplexity": 10,
        "codeCheckerFlake8MaxComplexityEnabled": True,
        "codeCheckerFlake8MaxLineLength": 79,
        "codeCheckerFlake8OtherOptions": "",
        "codeCheckerFlake8SelectEnabled": False,
        "codeCheckerFlake8SelectValues": "",
        "codeCheckerFlake8WarningNumbers": "E, C"}

import __main__
__main__.__dict__["_CHECKER_CACHE"] = _CHECKER_CACHE
###############################################################


class ItomFlakesReporter():
    """Formats the results of pyflakes checks and then presents them to the user."""
    def __init__(self, filename: str, lineOffset: int = 0, defaultMsgType: int = 2):
        self._items = []
        self._filename = filename
        self._lineOffset = lineOffset
        self._defaultMsgType = defaultMsgType
    
    def _addItem(self, type: int, filename: str, msgCode: str, description: str, lineNo: int = -1, column: int = -1):
        '''
        @param type: the type of message (0: Info, 1: Warning, 2: Error)
        @ptype type: C{int}
        '''
        self._items.append("%i::%s::%i::%i::%s::%s" % 
                            (type, self._filename, lineNo - self._lineOffset, 
                            column, msgCode, description))

    def unexpectedError(self, filename, msg):
        """
        An unexpected error occurred trying to process C{filename}.
        
        @param filename: The path to a file that we could not process.
        @ptype filename: C{unicode}
        @param msg: A message explaining the problem.
        @ptype msg: C{unicode}
        
        This method is called by pyflakes
        """
        self._addItem(type=2, filename=filename, msgCode="", description=msg, lineNo=-1, column=-1)
    
    def syntaxError(self, filename, msg, lineno, offset, text):
        """
        There was a syntax error in C{filename}.
        
        @param filename: The path to the file with the syntax error.
        @ptype filename: C{unicode}
        @param msg: An explanation of the syntax error.
        @ptype msg: C{unicode}
        @param lineno: The line number where the syntax error occurred.
        @ptype lineno: C{int}
        @param offset: The column on which the syntax error occurred, or None.
        @ptype offset: C{int}
        @param text: The source code containing the syntax error.
        @ptype text: C{unicode}
        
        This method is called by pyflakes
        """
        line = text.splitlines()[-1]
        
        if offset is not None:
            offset = offset - (len(text) - len(line))
            self._addItem(type=2, filename=filename, msgCode="", 
                            description=msg, lineNo=lineno, column=offset + 1)
        else:
            self._addItem(type=2, filename=filename, msgCode="", 
                            description=msg, lineNo=lineno, column=-1)

    def flake(self, message):
        """
        pyflakes found something wrong with the code.
        
        @param: A messages.Message.
        """
        msg = message.message % message.message_args
        self._addItem(type=self._defaultMsgType, filename=message.filename, msgCode="", 
                    description=msg, lineNo=message.lineno, column=message.col)
    
    def results(self):
        """
        returns a list of reported items.
        Every item is a string that can be separated by :: into 6 parts.
        The parts are: 
        1. type (int): 0: Info, 1: Warning, 2: Error
        2. filename (str): filename of tested file
        3. lineNo (int): line number or -1 if no line number
        4. columnIndex (int): column index or -1 if unknown
        5. message code (str): e.g. E550...
        6. description (str): text of error
        """
        return self._items

#############################################################


class ItomFlake8Formatter(base.BaseFormatter):
    """Print absolutely nothing."""
    
    def __init__(self, *args, **kwargs):
        super(ItomFlake8Formatter, self).__init__(*args, **kwargs)
        self._items = []
        self._line_offset: int = 0
    
    @property
    def line_offset(self) -> int:
        return self._line_offset
    
    @line_offset.setter
    def line_offset(self, value: int) -> int:
        self._line_offset = value
    
    def _addItem(self, errorType: int, filename: str, msgCode: str, description: str, lineNo: int = -1, column: int = -1):
        '''
        @param type: the type of message (0: Info, 1: Warning, 2: Error)
        @ptype type: C{int}
        '''
        self._items.append("%i::%s::%i::%i::%s::%s" % (errorType, filename, lineNo - self._line_offset, column, msgCode, description))
    
    def results(self):
        """
        returns a list of reported items.
        Every item is a string that can be separated by :: into 6 parts.
        The parts are: 
        1. type (int): 0: Info, 1: Warning, 2: Error
        2. filename (str): filename of tested file
        3. lineNo (int): line number or -1 if no line number
        4. columnIndex (int): column index or -1 if unknown
        5. message code (str): e.g. E550...
        6. description (str): text of error
        """
        return self._items
    
    def after_init(self):  # type: () -> None
        """Initialize the formatter further."""
        self._items = []

    def beginning(self, filename):  # type: (str) -> None
        """Notify the formatter that we're starting to process a file.

        :param str filename:
            The name of the file that Flake8 is beginning to report results
            from.
        """
        pass

    def finished(self, filename):  # type: (str) -> None
        """Notify the formatter that we've finished processing a file.

        :param str filename:
            The name of the file that Flake8 has finished reporting results
            from.
        """
        pass

    def start(self):  # type: () -> None
        """Prepare the formatter to receive input.

        This defaults to initializing :attr:`output_fd` if :attr:`filename`
        """
        pass

    def handle(self, error):  # type: (Violation) -> None
        """Handle an error reported by Flake8.

        This defaults to calling :meth:`format`, :meth:`show_source`, and
        then :meth:`write`. To extend how errors are handled, override this
        method.

        :param error:
            This will be an instance of
            :class:`~flake8.style_guide.Violation`.
        :type error:
            flake8.style_guide.Violation
        
        1. type (int): 0: Info, 1: Warning, 2: Error
        2. filename (str): filename of tested file
        3. lineNo (int): line number or -1 if no line number
        4. columnIndex (int): column index or -1 if unknown
        5. message code (str): e.g. E550...
        6. description (str): text of error
        """
        if error.code.startswith("F"):  # errors from pyflakes
            errorType = 2  # error
        elif error.code.startswith("E"):  # usually errors from pycodestyle
            errorType = 1  # warning
        else:  # e.g. "W" --> warnings from pycodestyle
            errorType = 0  # info
        
        self._addItem(errorType, error.filename, error.code, error.text, error.line_number, error.column_number)

#############################################################


def hasPyFlakes() -> bool:
    return _HAS_PYFLAKES


def hasFlake8() -> bool:
    return _HAS_FLAKE8



def check(codestring: str, 
          filename: str, 
          fileSaved: bool,
          mode: int = 1,
          autoImportItom: bool = False,
          furtherPropertiesJson : str = {}) -> List[str]:
    '''run the test for a single file.
    
    Args:
        codestring: is the code to be checked. 
            This code can be different from the current code in filename.
        filename: the filename of the code (must only exist if fileSaved is True)
        fileSaved: True if the filename contains the real code to be checked, else False
        mode: if 0: no code check is executed, if 1: only pyflakes is called, if 2: flake8 is called.
            The mode value must be equal to the enumeration ito::PythonCommon::CodeCheckerMode in the C++ code.
        autoImportItom: If True, a line 'from itom import dataObject, plot1, ...' is
            hiddenly added before the first line of the script.
        furtherPropertiesJson: further properties for the called modules as json string.
    
    Returns:
        results: This is a list of errors or other observations. Each error has the following
            format: <type:int>::<filename:str>::<lineNo::int>::<column:int>::<code::str>::<description::str>
            where <name:type> is a replacement for one value (with the given type).
            The type is 0: information, 1: warning, 2: error
            The filename is equal to the given filename for this file.
            The lineNo starts with 1 for the first line.
            The code are the original flake8 codes (e.g. W804) or an empty string for pyflakes calls.
    '''
    global _CHECKER_CACHE
    global _PUBLIC_ITOM_MODULES
    global _CONFIG_DEFAULTS
    
    propertiesChanged : bool = False
    config : dict = {}
    
    print("check(", filename, fileSaved, mode, autoImportItom, furtherPropertiesJson, ")")
    
    if not "importItomString" in _CHECKER_CACHE:
        _CHECKER_CACHE["importItomString"] = "from itom import %s" % ", ".join(_PUBLIC_ITOM_MODULES)
    
    if not "propertiesString" in _CHECKER_CACHE:
        _CHECKER_CACHE["propertiesString"] = furtherPropertiesJson
        _CHECKER_CACHE["properties"] = {} #will be read later
        propertiesChanged = True
    else:
        if _CHECKER_CACHE["propertiesString"] != furtherPropertiesJson:
            _CHECKER_CACHE["propertiesString"] = furtherPropertiesJson
            propertiesChanged = True
    
    if propertiesChanged:
        #update config with default values
        config = _CONFIG_DEFAULTS
        config.update(json.loads(_CHECKER_CACHE["propertiesString"]))
        _CHECKER_CACHE["properties"] = config
    
    config = _CHECKER_CACHE["properties"]
    
    if mode == 0: #NoCodeChecker
        return []
    
    elif mode == 1: #CodeCheckerPyFlakes
        if _HAS_PYFLAKES:
            if autoImportItom:
                codestring = "%s\n%s" % (_CHECKER_CACHE["importItomString"], codestring)
                lineOffset = 1
            else:
                lineOffset = 0
            
            reporter = ItomFlakesReporter(filename, 
                                          lineOffset = lineOffset, 
                                          defaultMsgType = config["codeCheckerPyFlakesCategory"])
            pyflakesapi.check(codestring, "code", reporter=reporter)
            return reporter.results()
        else:
            raise RuntimeError("Code check not possible, since module pyflakes missing")
    
    elif mode == 2: #CodeCheckerFlake8
        if _HAS_FLAKE8:
            if autoImportItom:
                codestring = "%s\n%s" % (_CHECKER_CACHE["importItomString"], codestring)
                lineOffset = 1
                fileSaved = False
            else:
                lineOffset = 0
            
            style_guide = flake8.get_style_guide()
            style_guide.init_report(reporter=ItomFlake8Formatter)
            reporter = style_guide._application.formatter #instance to ItomFlake8Formatter
            reporter.line_offset = lineOffset
            report = None
            
            if fileSaved:
                report = style_guide.check_files([filename, ])
            else:
                with tempfile.NamedTemporaryFile("wt", delete=False, suffix=".py") as fp:
                    tempfilename = fp.name
                    fp.write(codestring)
                
                try:
                    report = style_guide.check_files([tempfilename, ])
                    
                except Exception:
                    pass
                finally:
                    os.remove(tempfilename)
            
            if report is not None:
                results = reporter.results()
                # print("Run flake8 on file %s: %i" % (filename, report.total_errors))
                return results
            else:
                return []
        else:
            raise RuntimeError("Code check not possible, since module flake8 missing")
    
    else:
        raise RuntimeError("Code checker: invalid mode %i. Only 0, 1 or 2 supported" % mode)