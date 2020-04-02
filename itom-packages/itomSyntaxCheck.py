"""This module builds a bridge between the syntax and style checks
of itom and different possible packages in Python

Currently this script supports the Python packages:

* pyflakes
* flake8
"""

import tempfile
import os
from typing import List, Optional
import json
import re
import itom
import warnings
from builtins import filter as bfilter  # due to potential conflict with itom.filter
import time

try:
    from flake8.api import legacy as flake8legacy
    _HAS_FLAKE8: bool = True
except ModuleNotFoundError:
    _HAS_FLAKE8: bool = False

if _HAS_FLAKE8:
    import flake8
    from flake8.formatting import base
    from flake8.main import application as app
    from flake8.options import config
    
    if False:  # `typing.TYPE_CHECKING` was introduced in 3.5.2
        from flake8.style_guide import Violation

try:
    from pyflakes import api as pyflakesapi
    _HAS_PYFLAKES: bool = True
except ModuleNotFoundError:
    _HAS_PYFLAKES: bool = False


_PUBLIC_ITOM_MODULES = [item for item in dir(itom) if not item.startswith("__")]

_CHECKER_CACHE = {}  # cache to temporarily store further information

_CONFIG_DEFAULTS = {"codeCheckerPyFlakesCategory": 2,
                    "codeCheckerFlake8Docstyle": "pep257",
                    "codeCheckerFlake8ErrorNumbers": "F",
                    "codeCheckerFlake8IgnoreEnabled": True,
                    "codeCheckerFlake8IgnoreValues": "E121,E123,E126,E226,E24,E704,W503,W504,W293",
                    "codeCheckerFlake8MaxComplexity": 10,
                    "codeCheckerFlake8MaxComplexityEnabled": True,
                    "codeCheckerFlake8MaxLineLength": 79,
                    "codeCheckerFlake8OtherOptions": "",
                    "codeCheckerFlake8SelectEnabled": False,
                    "codeCheckerFlake8SelectValues": "",
                    "codeCheckerFlake8WarningNumbers": "E, C"}

#############################################################
class TimeIt(object):
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]: %.4f s elapsed' % (self.name, time.time() - self.tstart))
        else:
            print('%.4f s elapsed' % (time.time() - self.tstart))


def _checkErrorCodeStringList(text: str) -> Optional[str]:
    """Checks the given text for a valid code string list format.
    
    Every single code string must be a letter a-z or A-Z
    followed by 0-4 numbers (e.g. E999).
    
    The text can be zero, one or multiple code strings, joint
    by a comma (surrounding spaces are allowed). E.g:
    
    W921, e23,F
    
    Args:
        text: the given error code string
    
    Returns:
        corrected: None if the text does not fit or the corrected
            text where all letters are turned to capital letters
            and spaces are removed.
    """
    validName = \
        re.compile(r"^([A-Za-z]([0-9]){0,4}\s*,\s*)*([A-Za-z]([0-9]){0,4})?\s*$")
    
    if validName.match(text):
        return text.replace(" ", "").upper()
    else:
        return None
    
#############################################################


class CheckerWarning(Warning):
    """warning raised if any parameter to the checkers is invalid.
    This warning should always be presented to itom users.
    """
    pass

#############################################################


class ItomFlakesReporter():
    """Formats the results of pyflakes checks and then presents them to the user."""
    def __init__(self, filename: str, lineOffset: int = 0, defaultMsgType: int = 2):
        self._items = []
        self._filename = filename
        self._lineOffset = lineOffset
        self._defaultMsgType = defaultMsgType
    
    def reset(self):
        """reset all current message items
        """
        self._items = []
    
    def _addItem(self, type: int, filename: str, msgCode: str, description: str, lineNo: int = -1, column: int = -1):
        """
        @param type: the type of message (0: Info, 1: Warning, 2: Error)
        @ptype type: C{int}
        """
        if lineNo > self._lineOffset:
            # all messages earlier than _lineOffset belong to the additional itom import.
            # ignore them.
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
        self._lineOffset: int = 0
        self._errorCodes: str = ["F", ]
        self._warningCodes: str = ["E", "C"]
    
    @property
    def line_offset(self) -> int:
        return self._lineOffset
    
    @line_offset.setter
    def line_offset(self, value: int) -> int:
        self._lineOffset = value
    
    def _check_categories(self, codes: str, error: bool) -> List[str]:
        """
        """
        codeCorrected = _checkErrorCodeStringList(codes)
        
        if codeCorrected:
            return codeCorrected.split(",")
        else:
            warnings.warn("invalid code name: %s" % codes, CheckerWarning)
            if error:
                return ["F", ]
            else:
                return ["E", "C"]
    
    def set_warn_and_error_categories(self, errorCodes: str, warnCodes: str):
        """
        """
        self._errorCodes = self._check_categories(errorCodes, True)
        self._warningCodes = self._check_categories(warnCodes, False)
    
    def _addItem(self, errorType: int,
                 filename: str,
                 msgCode: str,
                 description: str,
                 lineNo: int = -1,
                 column: int = -1):
        """
        @param type: the type of message (0: Info, 1: Warning, 2: Error)
        @ptype type: C{int}
        """
        if lineNo > self._lineOffset:
            # all messages earlier than _lineOffset belong to the additional itom import.
            # ignore them.
            self._items.append("%i::%s::%i::%i::%s::%s" % 
                               (errorType, filename, 
                                lineNo - self._lineOffset, column, 
                                msgCode, description))
    
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
        
        # error code E999 is a real syntax error
        if error.code == "E999" or \
           list(bfilter(error.code.startswith, self._errorCodes)) != []:
            errorType = 2  # error
        elif list(bfilter(error.code.startswith, self._warningCodes)) != []:
            errorType = 1  # warning
        else:  # e.g. "W" --> warnings from pycodestyle
            errorType = 0  # info
        
        self._addItem(errorType,
                      error.filename,
                      error.code,
                      error.text,
                      error.line_number,
                      error.column_number)


def hasPyFlakes() -> bool:
    return _HAS_PYFLAKES


def hasFlake8() -> bool:
    return _HAS_FLAKE8


def flake8GetStyleGuideItom(base_directory, **kwargs):
    r"""Provision a StyleGuide for use.
    
    This is a rewritten method of flake8/api/legacy.py.
    
    Originally, in flake8, the option overload order is (the latter wins):
    
    1. flake8 defaults
    2. optional tox.ini, .flake8 etc. configuration files
    3. arguments passed in kwargs (which are the settings from the itom
            property dialog)
    
    However, the desired behaviour is:
    
    1. flake8 defaults
    2. arguments passed in kwargs
    3. optional tox.ini, .flake8 etc. configuration files.
    
    :param base_directory:
        locally change the current working dir to base_directory, such that
        local config files are properly found. Reset it afterwards.
    :param \*\*kwargs:
        Keyword arguments that provide some options for the StyleGuide.
    :returns:
        An initialized StyleGuide
    :rtype:
        :class:`StyleGuide`
    """
    current_working_dir = os.getcwd()
    os.chdir(base_directory)
    
    application = app.Application()
    application.parse_preliminary_options_and_args([])  # for max logging pass: ["-vvv"]
    flake8.configure_logging(
        application.prelim_opts.verbose, application.prelim_opts.output_file
    )
    application.make_config_finder()
    
    application.find_plugins()
    application.register_plugin_options()
    application.parse_configuration_and_cli([])
    
    config_parser = config.MergedConfigParser(
        option_manager=application.option_manager,
        config_finder=application.config_finder
    )
    
    # Get the local (project) config again
    local_config = config_parser.parse_local_config()
    
    # the dependent local config files can be read by this list:
    local_config_files = application.config_finder._local_found_files  # careful: might change!!!
    # print("local_files:", local_config_files)
    
    # reset current working directory
    os.chdir(current_working_dir)
    
    # We basically want application.initialize to be called but with these
    # options set instead before we make our formatter, notifier, internal
    # style guide and file checker manager.
    
    # itom specific: only update the option if not contained in local_config_files.
    # The local config file should be more important than the itom settings.
    options = application.options
    for key, value in kwargs.items():
        if not key in local_config:
            try:
                getattr(options, key)
                setattr(options, key, value)
            except AttributeError:
                pass
                # LOG.error('Could not update option "%s"', key)
    application.make_formatter()
    application.make_guide()
    application.make_file_checker_manager()
    return flake8legacy.StyleGuide(application)


def createFlake8OptionsFromProperties(props: dict) -> dict:
    """converts properties, obtained from itom, to a options dict,
    that can be passed to flake8.
    """
    options = {}
    errors = []
    
    #print("props", str(props))
    
    # if pydocstyle is installed:
    if "codeCheckerFlake8Docstyle" in props:
        options["docstring_convention"] = props["codeCheckerFlake8Docstyle"]
    
    if "codeCheckerFlake8IgnoreEnabled" in props and \
       props["codeCheckerFlake8IgnoreEnabled"]:
        values = _checkErrorCodeStringList(props["codeCheckerFlake8IgnoreValues"])
        
        if values is not None:
            options["ignore"] = values
        else:
            errors.append(
                "ignore values of 'flake8' code checker "
                "are invalid. They will be ignored.")
    
    if "codeCheckerFlake8SelectEnabled" in props and \
       props["codeCheckerFlake8SelectEnabled"]:
        values = _checkErrorCodeStringList(props["codeCheckerFlake8SelectValues"])
        
        if values is not None:
            options["select"] = values
        else:
            errors.append(
                "select values of 'flake8' code checker "
                "are invalid. They will be ignored.")
    
    if "codeCheckerFlake8MaxLineLength" in props:
        options["max_line_length"] = props["codeCheckerFlake8MaxLineLength"]
    
    if "codeCheckerFlake8MaxComplexityEnabled" in props and \
       props["codeCheckerFlake8MaxComplexityEnabled"]:
        options["max_complexity"] = props["codeCheckerFlake8MaxComplexity"]
    
    if "codeCheckerFlake8OtherOptions" in props and \
       props["codeCheckerFlake8OtherOptions"] != "":
        lines = props["codeCheckerFlake8OtherOptions"].replace("\r", "").split("\n")
        
        for line in lines:
            parts = line.split("=")
            
            if len(parts) != 2:
                errors.append("additional option line '%s' of "
                              "'flake8' code checker has an invalid format "
                              "(option=value)" % line)
            else:
                name = parts[0].strip()
                name = name.replace("-", "_")
                value = parts[1].strip()
                
                if len(name) == 0 or len(value) == 0:
                    errors.append("additional option line '%s' of "
                                  "'flake8' code checker has an invalid format "
                                  "(option=value)" % line)
                else:
                    options[name] = value
    
    # print(str(options))
    
    if len(errors) > 0:
        warnings.warn(";".join(errors), CheckerWarning)
    
    return options


def check(codestring: str, 
          filename: str, 
          fileSaved: bool,
          mode: int = 1,
          autoImportItom: bool = False,
          furtherPropertiesJson: str = {}) -> List[str]:
    """run the test for a single file.
    
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
        results: This is a list of errors or other observations. 
            Each error has the following format: <type:int>::<filename:str>::
                <lineNo::int>::<column:int>::<code::str>::<description::str>
            
            where <name:type> is a replacement for one value 
            (with the given type). 
            The type is 0: information, 1: warning, 2: error
            The filename is equal to the given filename for this file.
            The lineNo starts with 1 for the first line.
            The code are the original flake8 codes (e.g. W804) or an empty 
            string for pyflakes calls.
    """
    
    global _CHECKER_CACHE
    global _PUBLIC_ITOM_MODULES
    global _CONFIG_DEFAULTS
    
    propertiesChanged: bool = False
    config: dict = {}
    
    # print("check(", filename, fileSaved, mode, autoImportItom, furtherPropertiesJson, ")")
    
    if "importItomString" not in _CHECKER_CACHE:
        _CHECKER_CACHE["importItomString"] = \
            "from itom import %s" % ", ".join(_PUBLIC_ITOM_MODULES)
    
    if "propertiesString" not in _CHECKER_CACHE:
        _CHECKER_CACHE["propertiesString"] = furtherPropertiesJson
        _CHECKER_CACHE["properties"] = {}  # will be read later
        _CHECKER_CACHE["flake8options"] = {}  # will be read later
        propertiesChanged = True
    else:
        if _CHECKER_CACHE["propertiesString"] != furtherPropertiesJson:
            _CHECKER_CACHE["propertiesString"] = furtherPropertiesJson
            propertiesChanged = True
    
    if propertiesChanged:
        # update config with default values
        config = _CONFIG_DEFAULTS
        config.update(json.loads(_CHECKER_CACHE["propertiesString"]))
        _CHECKER_CACHE["properties"] = config
        
        _CHECKER_CACHE["flake8options"] = \
            createFlake8OptionsFromProperties(config)
    
    config = _CHECKER_CACHE["properties"]
    
    if mode == 0:  # NoCodeChecker
        return []
    
    elif mode == 1:  # CodeCheckerPyFlakes
        if _HAS_PYFLAKES:
            if autoImportItom:
                codestring = "%s\n%s" % (_CHECKER_CACHE["importItomString"], codestring)
                lineOffset = 1
            else:
                lineOffset = 0
            
            reporter = ItomFlakesReporter(filename,
                                          lineOffset=lineOffset,
                                          defaultMsgType=config["codeCheckerPyFlakesCategory"])
            pyflakesapi.check(codestring, "code", reporter=reporter)
            return reporter.results()
        else:
            raise RuntimeError("Code check not possible, since module pyflakes missing")
    
    elif mode == 2:  # CodeCheckerFlake8
        if _HAS_FLAKE8:
            if autoImportItom:
                codestring = "%s\n%s" % (_CHECKER_CACHE["importItomString"], codestring)
                lineOffset = 1
                fileSaved = False
            else:
                lineOffset = 0
            
            #with TimeIt("flake8 loader"):
            baseDir = os.path.abspath(os.path.dirname(filename))
            style_guide = flake8GetStyleGuideItom(baseDir, **_CHECKER_CACHE["flake8options"])
            
            style_guide.init_report(reporter=ItomFlake8Formatter)
            reporter = style_guide._application.formatter  # instance to ItomFlake8Formatter
            reporter.line_offset = lineOffset
            reporter.set_warn_and_error_categories(
                config["codeCheckerFlake8ErrorNumbers"],
                config["codeCheckerFlake8WarningNumbers"])
            report = None
            
            if fileSaved:
                with warnings.catch_warnings():
                    # when parsing the file by the checker, a warning
                    # can occure (e.g. from an assert statement). ignore this warning.
                    warnings.simplefilter("ignore")
                    report = style_guide.check_files([filename, ])
            else:
                with tempfile.NamedTemporaryFile("wt", delete=False, suffix=".py") as fp:
                    tempfilename = fp.name
                    fp.write(codestring)
                
                try:
                    with warnings.catch_warnings():
                        # when parsing the file by the checker, a warning
                        # can occure (e.g. from an assert statement). ignore this warning.
                        warnings.simplefilter("ignore")
                        report = style_guide.check_files([tempfilename, ])
                except Exception as ex:
                    # import traceback
                    # traceback.print_exc()
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


if __name__ == "__main__":
    codestring = """def test(i : str , b) :
    a=2*3
    assert(1, 1)
    
    return a"""
    
    filename = "temp.py"
    fileSaved = False
    mode = 2  # all checks
    autoImportItom = False
    furtherPropertiesJson = """{
    "codeCheckerFlake8Docstyle": "google",
    "codeCheckerFlake8ErrorNumbers": "F",
    "codeCheckerFlake8IgnoreEnabled": true,
    "codeCheckerFlake8IgnoreValues": "W293",
    "codeCheckerFlake8MaxComplexity": 9,
    "codeCheckerFlake8MaxComplexityEnabled": true,
    "codeCheckerFlake8MaxLineLength": 81,
    "codeCheckerFlake8OtherOptions": "",
    "codeCheckerFlake8SelectEnabled": false,
    "codeCheckerFlake8SelectValues": "",
    "codeCheckerFlake8WarningNumbers": "E, C",
    "codeCheckerPyFlakesCategory": 1
}"""
    
    result = check(codestring, filename, fileSaved, mode, autoImportItom, furtherPropertiesJson)
    
    import pprint
    pprint.pprint(result)
