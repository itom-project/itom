"""This module builds a bridge between the syntax and style checks
of itom and different possible packages in Python

Currently this script supports the Python packages:

* pyflakes
* flake8

License information:

itom software
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2020, Institut fuer Technische Optik (ITO),
Universitaet Stuttgart, Germany

This file is part of itom.

itom is free software; you can redistribute it and/or modify it
under the terms of the GNU Library General Public Licence as published by
the Free Software Foundation; either version 2 of the Licence, or (at
your option) any later version.

itom is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
General Public Licence for more details.

You should have received a copy of the GNU Library General Public License
along with itom. If not, see <http://www.gnu.org/licenses/>.
"""

import tempfile
import os
import json
import re
import itom
import warnings
from builtins import filter as bfilter  # due to potential conflict with itom.filter
import time
import logging

try:
    from flake8.api import legacy as flake8legacy

    _HAS_FLAKE8 = True
except ImportError:
    _HAS_FLAKE8 = False

try:
    import flake8_docstrings
    _HAS_FLAKE8_DOCSTRINGS = True
except ImportError:
    _HAS_FLAKE8_DOCSTRINGS = False

if _HAS_FLAKE8:
    import flake8
    from flake8.formatting import base
    from flake8.main import application as app
    from flake8.options import config

    if False:  # `typing.TYPE_CHECKING` was introduced in 3.5.2
        from flake8.style_guide import Violation

    if flake8.__version__ >= "6":
        from flake8.options.parse_args import parse_args

    # disable the flake8.checker logger
    log = logging.getLogger("flake8.checker")
    log.disabled = True
    # disable the logger for warnings in the options manager of flake8
    log = logging.getLogger("flake8.options.manager")
    log.disabled = True
try:
    from pyflakes import api as pyflakesapi

    _HAS_PYFLAKES = True
except ModuleNotFoundError:
    _HAS_PYFLAKES = False

# all public modules, methods and classes of itom
_PUBLIC_ITOM_MODULES = [item for item in dir(itom) if not item.startswith("__")]

# cache to temporarily store further information
_CHECKER_CACHE = {}

# default configuration values from itom property page.
# This dictionary must be kept in sync with widgetPropEditorCodeCheckers.cpp
_CONFIG_DEFAULTS = {
    "codeCheckerPyFlakesCategory": 2,
    "codeCheckerFlake8Docstyle": "pep257",
    "codeCheckerFlake8ErrorNumbers": "F",
    "codeCheckerFlake8IgnoreEnabled": False,
    "codeCheckerFlake8IgnoreValues": "",
    "codeCheckerFlake8IgnoreExtendEnabled": True,
    "codeCheckerFlake8IgnoreExtendValues": "W293",
    "codeCheckerFlake8MaxComplexity": 10,
    "codeCheckerFlake8MaxComplexityEnabled": True,
    "codeCheckerFlake8MaxLineLength": 79,
    "codeCheckerFlake8OtherOptions": "",
    "codeCheckerFlake8SelectEnabled": False,
    "codeCheckerFlake8SelectValues": "",
    "codeCheckerFlake8WarningNumbers": "E, C",
}


class TimeIt(object):
    """Time measurement for a code block.

    This class is only necessary for debug purposes and
    provides a context manager.

    Example:
        with TimeIt("Step1"):
            doSomething(...)
    """

    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print("[%s]: %.4f s elapsed" % (self.name, time.time() - self.tstart))
        else:
            print("%.4f s elapsed" % (time.time() - self.tstart))


def _checkErrorCodeStringList(text: str):
    """Checks the given text for a valid code string list format.

    Every single code string must be a letter a-z or A-Z
    followed by 0-4 numbers (e.g. E999).

    The text can be zero, one or multiple code strings, joint
    by a comma (surrounding spaces are allowed). E.g:

    W921, e23,F

    Args:
        text (str): the given error code string

    Returns:
        Optional[str]: None if the text does not fit or the corrected
            text where all letters are turned to capital letters
            and spaces are removed.
    """
    validName = re.compile(
        r"^([A-Za-z]([0-9]){0,4}\s*,\s*)*([A-Za-z]([0-9]){0,4})?\s*$"
    )

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


class ItomFlakesReporter:
    """Formats the results of pyflakes checks to be consumed by itom.

    This class provides the interface such that itom (pythonEngine) can
    read the results from a pyflakes file check.
    """

    def __init__(self, filename, lineOffset=0, defaultMsgType=2):
        """Constructor.

        Args:
            filename (str): the filename that has been checked.
            lineOffset (int): if a message is reported, the lineOffset is subtracted to the line
                number before passing to itom (used to ignore the additional import
                statement, that is added if the itom module should automatically be
                considered to be auto-imported).
            defaultMsgType (int): all messages beside syntax errors will obtain this message type.
                0: info, 1: warning, 2: error
        """
        self._items = []
        self._filename = filename
        self._lineOffset = lineOffset
        self._defaultMsgType = defaultMsgType

    def reset(self):
        """Reset all current message items."""
        self._items = []

    def _addItem(self, msgType, filename, msgCode, description, lineNo=-1, column=-1):
        """Internal method to add a new item to the list of items.

        Args:
            type (int): the type of the message (0: info, 1: warning, 2: error)
            filename (str): the filename of the message
            msgCode (str): the pyflakes message code (usually an empty string)
            description (str): the message text
            lineNo (int): the line number in the checked file that is the
                reason for the message
            column (int): the column index of the start of the error or -1 if unknown
        """
        assert msgType in [0, 1, 2]

        if lineNo > self._lineOffset:
            # all messages earlier than _lineOffset belong
            # to the additional itom import. Ignore them.
            self._items.append(
                "%i::%s::%i::%i::%s::%s"
                % (
                    msgType,
                    self._filename,
                    lineNo - self._lineOffset,
                    column,
                    msgCode,
                    description,
                )
            )

    def unexpectedError(self, filename, msg):
        """An unexpected error occurred trying to process C{filename}.

        This method is called by pyflakes

        Args:
            filename (str): The path to a file that we could not process.
            msg (str): A message explaining the problem.
        """
        self._addItem(
            msgType=2,
            filename=filename,
            msgCode="",
            description=msg,
            lineNo=-1,
            column=-1,
        )

    def syntaxError(self, filename, msg, lineno, offset, text):
        """
        There was a syntax error in C{filename}.

        Args:
           filename (str): The path to the file with the syntax error.
           msg (str): An explanation of the syntax error.
           lineno (int): The line number where the syntax error occurred.
           offset (Optional[int]): The column on which the syntax error occurred, or None.
           text (str): The source code containing the syntax error.

        This method is called by pyflakes
        """
        line = text.splitlines()[-1]

        if offset is not None:
            offset = offset - (len(text) - len(line))
            self._addItem(
                msgType=2,
                filename=filename,
                msgCode="",
                description=msg,
                lineNo=lineno,
                column=offset + 1,
            )
        else:
            self._addItem(
                msgType=2,
                filename=filename,
                msgCode="",
                description=msg,
                lineNo=lineno,
                column=-1,
            )

    def flake(self, message):
        """pyflakes found something wrong with the code.

        Arg: A messages.Message. message.col is the index of the column,
            where the indication of an error etc. starts. message.lineno
            is the line number of the indication
            (starting with 1 for the first line).
        """
        msg = message.message % message.message_args
        self._addItem(
            msgType=self._defaultMsgType,
            filename=message.filename,
            msgCode="",
            description=msg,
            lineNo=message.lineno,
            column=message.col,
        )

    def results(self):
        """Called by pythonEngine in itom to obtain the
        current list of reported items.

        Every item is a string that can be separated by '::' into 6 parts.
        The parts are:

        1. type (int): 0: Info, 1: Warning, 2: Error
        2. filename (str): filename of tested file
        3. lineNo (int): line number or -1 if no line number
        4. columnIndex (int): column index or -1 if unknown
        5. message code (str): e.g. E550...
        6. description (str): text of error
        """
        return self._items


if _HAS_FLAKE8:
    # code in this section is only active if flake8 is available.

    class ItomFlake8Formatter(base.BaseFormatter):
        """Special formatter class to flake8 for itom.

        This class prevents any printed output of flake8
        observations. Instead all messages are collected
        in a list of serialized message information. This list
        is the read by itom in order to visualize it in the
        editor windows.
        """

        def __init__(self, *args, **kwargs):
            """Constructor."""
            super(ItomFlake8Formatter, self).__init__(*args, **kwargs)
            self._items = []
            self._errorCodes = [
                "F",
            ]
            self._warningCodes = ["E", "C"]

        def _check_categories(self, codes, error):
            """
            Args:
                codes (str) :
                error (bool) :

            Returns:
                List[str] :
            """
            codeCorrected = _checkErrorCodeStringList(codes)

            if codeCorrected:
                return codeCorrected.split(",")
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn("invalid code name: %s" % codes, CheckerWarning)

                if error:
                    return [
                        "F",
                    ]
                else:
                    return ["E", "C"]

        def set_warn_and_error_categories(self, errorCodes, warnCodes):
            """
            Args:
                errorCodes (str) :
                warnCodes (str) :
            """
            self._errorCodes = self._check_categories(errorCodes, True)
            self._warningCodes = self._check_categories(warnCodes, False)

        def _addItem(
            self, errorType, filename, msgCode, description, lineNo=-1, column=-1,
        ):
            """
               type: the type of message (0: Info, 1: Warning, 2: Error)
            @ptype type: C{int}

            Args:
                errorType (int) :
                filename (str) :
                msgCode (str) :
                description (str) :
                lineNo (int) :
                column (int) :
            """
            if lineNo > 0:
                self._items.append(
                    "%i::%s::%i::%i::%s::%s"
                    % (errorType, filename, lineNo, column, msgCode, description,)
                )

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
            if (
                error.code == "E999"
                or error.code == "E902"
                or list(bfilter(error.code.startswith, self._errorCodes)) != []
            ):
                errorType = 2  # error
            elif list(bfilter(error.code.startswith, self._warningCodes)) != []:
                errorType = 1  # warning
            else:  # e.g. "W" --> warnings from pycodestyle
                errorType = 0  # info

            self._addItem(
                errorType,
                error.filename,
                error.code,
                error.text,
                error.line_number,
                error.column_number - 1,
            )

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

        if flake8.__version__ >= "6":
            # flake8 >= 6.0

            # get parameters from tox.ini, setup.cfg or .flake8 files
            cfg, cfg_dir = config.load_config(
                config=None,
                extra=[],
                isolated=False,
            )

            if "flake8" in cfg:
                # remove all parameters in cfg["flake8"] from kwargs
                # (settings from itom properties) to set the priority to cfg.
                for cfg_item in cfg["flake8"]:
                    cfg_item = cfg_item.replace("-", "_")
                    if cfg_item in kwargs:
                        # do not overwrite itom settings, that are set by any project
                        # setting file (tox.ini, setup.cfg ..)
                        del kwargs[cfg_item]

            kwargs_parsed = []

            for item in kwargs:
                val = kwargs[item]
                item = item.replace("_", "-")

                if type(val) is list:
                    kwargs_parsed.append(
                        "--%s=%s" % (item, ",".join([str(ii) for ii in val]))
                        )
                else:
                    kwargs_parsed.append("--%s=%s" % (item, val))

            application.plugins, application.options = parse_args(kwargs_parsed)

            # reset kwargs, since already handled.
            kwargs = {}
        elif flake8.__version__ >= "5":
            # flake8 >= 5.0
            prelim_opts, remaining_args = application.parse_preliminary_options([])
            flake8.configure_logging(prelim_opts.verbose, prelim_opts.output_file)

            cfg, cfg_dir = config.load_config(
                config=prelim_opts.config,
                extra=prelim_opts.append_config,
                isolated=prelim_opts.isolated,
            )

            application.find_plugins(
                cfg,
                cfg_dir,
                enable_extensions=prelim_opts.enable_extensions,
                require_plugins=prelim_opts.require_plugins,
            )

            application.register_plugin_options()
            application.parse_configuration_and_cli(cfg, cfg_dir, remaining_args)

            # Get the local (project, e.g. tox.ini) config again
            local_config = {}

            if "flake8" in cfg:
                local_config = dict(cfg["flake8"])

        elif not hasattr(
            application, "parse_preliminary_options_and_args"
        ):  # flake8 >= 3.8
            prelim_opts, remaining_args = application.parse_preliminary_options([])
            flake8.configure_logging(prelim_opts.verbose, prelim_opts.output_file)

            config_finder = config.ConfigFileFinder(
                application.program,
                prelim_opts.append_config,
                config_file=prelim_opts.config,
                ignore_config_files=prelim_opts.isolated,
            )

            application.find_plugins(config_finder)
            application.register_plugin_options()
            application.parse_configuration_and_cli(config_finder, remaining_args)

            if flake8.__version__ < "4.0.0":
                config_parser = config.MergedConfigParser(
                    option_manager=application.option_manager, config_finder=config_finder,
                )
            else:
                config_parser = config.ConfigParser(
                    option_manager=application.option_manager, config_finder=config_finder,
                )

            # Get the local (project, e.g. tox.ini) config again
            local_config = config_parser.parse_local_config()

        else:  # for older versions of flake8 < 3.8.0
            application.parse_preliminary_options_and_args(
                []
            )  # for max logging pass: ["-vvv"]

            flake8.configure_logging(
                application.prelim_opts.verbose, application.prelim_opts.output_file,
            )
            application.make_config_finder()

            application.find_plugins()
            application.register_plugin_options()

            application.parse_configuration_and_cli([])

            config_finder = application.config_finder

            config_parser = config.MergedConfigParser(
                option_manager=application.option_manager, config_finder=config_finder,
            )

            # Get the local (project, e.g. tox.ini) config again
            local_config = config_parser.parse_local_config()

        # the dependent local config files can be read by this list:
        # local_config_files = config_finder._local_found_files  # careful: might change!!!
        # print("local_files:", local_config_files)

        # reset current working directory
        os.chdir(current_working_dir)

        # We basically want application.initialize to be called but with these
        # options set instead before we make our formatter, notifier, internal
        # style guide and file checker manager.

        # itom specific: only update the option if not contained in local_config_files.
        # The local config file should be more important than the itom settings.
        options = application.options
        options_extended = False

        for key, value in kwargs.items():
            if key not in local_config:
                try:
                    getattr(options, key)
                    setattr(options, key, value)
                    options_extended = True
                except AttributeError:
                    pass
                    # LOG.error('Could not update option "%s"', key)

        if options_extended:
            # parse the options again, since they have been changed again
            # and must then be propagated to the flake8 plugins
            # the following lines are taken from
            # application.parse_configuration_and_cli

            if flake8.__version__ < "5.0":
                options._running_from_vcs = False

                application.check_plugins.provide_options(
                    application.option_manager, options, application.args
                )
                application.formatting_plugins.provide_options(
                    application.option_manager, options, application.args
                )
            else:
                for loaded in application.plugins.all_plugins():
                    parse_options = getattr(loaded.obj, "parse_options", None)
                    if parse_options is None:
                        continue

                    # XXX: ideally we wouldn't have two forms of parse_options
                    try:
                        parse_options(
                            application.option_manager,
                            options,
                            options.filenames,
                        )
                    except TypeError:
                        parse_options(options)

        application.make_formatter()
        application.make_guide()

        if flake8.__version__ >= "6":
            application.make_file_checker_manager([])
        else:
            application.make_file_checker_manager()
        return flake8legacy.StyleGuide(application)

    def createFlake8OptionsFromProperties(props):
        """converts properties, obtained from itom, to a options dict,
        that can be passed to flake8.

        Args:
            props (dict) :

        Returns:
            dict :
        """
        options = {}
        errors = []

        # if pydocstyle is installed:
        if "codeCheckerFlake8Docstyle" in props:
            if _HAS_FLAKE8_DOCSTRINGS:
                options["docstring_convention"] = props["codeCheckerFlake8Docstyle"]

        if (
            "codeCheckerFlake8IgnoreEnabled" in props
            and props["codeCheckerFlake8IgnoreEnabled"]
        ):
            values = _checkErrorCodeStringList(props["codeCheckerFlake8IgnoreValues"])

            if values is not None:
                options["ignore"] = values
            else:
                errors.append(
                    "ignore values of 'flake8' code checker "
                    "are invalid. They will be ignored."
                )

        if (
            "codeCheckerFlake8IgnoreExtendEnabled" in props
            and props["codeCheckerFlake8IgnoreExtendEnabled"]
        ):
            values = _checkErrorCodeStringList(
                props["codeCheckerFlake8IgnoreExtendValues"]
            )

            if values is not None:
                options["extend_ignore"] = values
            else:
                errors.append(
                    "extend_ingore values of 'flake8' code checker "
                    "are invalid. They will be ignored."
                )

        if (
            "codeCheckerFlake8SelectEnabled" in props
            and props["codeCheckerFlake8SelectEnabled"]
        ):
            values = _checkErrorCodeStringList(props["codeCheckerFlake8SelectValues"])

            if values is not None:
                options["select"] = values
            else:
                errors.append(
                    "select values of 'flake8' code checker "
                    "are invalid. They will be ignored."
                )

        if "codeCheckerFlake8MaxLineLength" in props:
            options["max_line_length"] = props["codeCheckerFlake8MaxLineLength"]

        if (
            "codeCheckerFlake8MaxComplexityEnabled" in props
            and props["codeCheckerFlake8MaxComplexityEnabled"]
        ):
            options["max_complexity"] = props["codeCheckerFlake8MaxComplexity"]

        if (
            "codeCheckerFlake8OtherOptions" in props
            and props["codeCheckerFlake8OtherOptions"] != ""
        ):
            lines = props["codeCheckerFlake8OtherOptions"].replace("\r", "").split("\n")

            for line in lines:
                parts = line.split("=")

                if line.strip() == "":
                    continue
                elif len(parts) == 1:
                    options[parts[0].strip()] = True
                elif len(parts) == 2:
                    name = parts[0].strip()
                    name = name.replace("-", "_")
                    value = parts[1].strip()

                    if len(name) == 0 or len(value) == 0:
                        errors.append(
                            "additional option line '%s' of "
                            "'flake8' code checker has an invalid format "
                            "(option=value)" % line
                        )
                    else:
                        # if value contains a comma, it is likely that this is
                        # a list of values: convert it...
                        if "," in value:
                            value = value.split(",")

                        options[name] = value
                else:
                    errors.append(
                        "additional option line '%s' of "
                        "'flake8' code checker has an invalid format "
                        "(option=value)" % line
                    )

        if len(errors) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(";".join(errors), CheckerWarning)

        return options


else:  # no flake8

    def createFlake8OptionsFromProperties(props):
        """This is a dummy method if flake8 not available.

        Args:
            props (dict) :

        Returns:
            dict :
        """
        return {}


def hasPyFlakes():
    """Return True if the package pyflakes is available, else False."""
    return _HAS_PYFLAKES


def hasFlake8():
    """Returns True if the package flake8 is available, else False."""
    return _HAS_FLAKE8


def check(
    codestring,
    filename,
    fileSaved,
    mode=1,
    autoImportItom=False,
    furtherPropertiesJson={},
):
    """Run the test for a single file.

    Args:
        codestring (str): is the code to be checked.
            This code can be different from the current code in filename.
        filename (str): the filename of the code
            (must only exist if fileSaved is True)
        fileSaved (bool): True if the filename contains the real code
            to be checked, else False
        mode (int): if 0: no code check is executed, if 1: only pyflakes
            is called, if 2: flake8 is called.
            The mode value must be equal to the enumeration
            ito::PythonCommon::CodeCheckerMode in the C++ code.
        autoImportItom (bool): If True, a line 'from itom import dataObject, plot1, ...' is
            hiddenly added before the first line of the script.
        furtherPropertiesJson (str): further properties for the called
            modules as json string.

    Returns:
        List[str]: This is a list of errors or other observations.
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

    propertiesChanged = False
    config = {}

    if "importItomString" not in _CHECKER_CACHE:
        _CHECKER_CACHE["importItomString"] = "from itom import %s" % ", ".join(
            _PUBLIC_ITOM_MODULES
        )

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

        _CHECKER_CACHE["flake8options"] = createFlake8OptionsFromProperties(config)

    config = _CHECKER_CACHE["properties"]

    if mode == 0:  # NoCodeChecker
        return []

    elif mode == 1:  # CodeCheckerPyFlakes
        if _HAS_PYFLAKES:
            if autoImportItom:
                codestring = "%s\n%s" % (
                    _CHECKER_CACHE["importItomString"],
                    codestring,
                )
                lineOffset = 1
            else:
                lineOffset = 0

            reporter = ItomFlakesReporter(
                filename,
                lineOffset=lineOffset,
                defaultMsgType=config["codeCheckerPyFlakesCategory"],
            )
            pyflakesapi.check(codestring, "code", reporter=reporter)
            return reporter.results()
        else:
            raise RuntimeError("Code check not possible, since module pyflakes missing")

    elif mode == 2:  # CodeCheckerFlake8
        if _HAS_FLAKE8:

            if autoImportItom:
                # add the itom imports as builtins option to flake8.
                # This overwrites other builtins, set in user or project config files!
                _CHECKER_CACHE["flake8options"]["builtins"] = _PUBLIC_ITOM_MODULES

            # with TimeIt("flake8 loader"):
            baseDir = os.path.abspath(os.path.dirname(filename))
            style_guide = flake8GetStyleGuideItom(
                baseDir, **_CHECKER_CACHE["flake8options"]
            )

            style_guide.init_report(reporter=ItomFlake8Formatter)
            reporter = (
                style_guide._application.formatter
            )  # instance to ItomFlake8Formatter
            reporter.set_warn_and_error_categories(
                config["codeCheckerFlake8ErrorNumbers"],
                config["codeCheckerFlake8WarningNumbers"],
            )
            report = None

            if fileSaved:
                # print("check saved file %s" % filename)
                with warnings.catch_warnings():
                    # when parsing the file by the checker, a warning
                    # can occure (e.g. from an assert statement). ignore this warning.
                    warnings.simplefilter("ignore")
                    try:
                        report = style_guide.check_files([filename,])
                    except Exception as ex:
                        # import traceback
                        # traceback.print_exc()
                        pass
            else:
                with tempfile.NamedTemporaryFile(
                    "wt", encoding="utf-8", suffix=".py", delete=False
                ) as fp:
                    tempfilename = fp.name
                    fp.write(codestring)
                    # print("check saved file %s" % tempfilename)
                try:
                    with warnings.catch_warnings():
                        # when parsing the file by the checker, a warning
                        # can occure (e.g. from an assert statement). ignore this warning.
                        warnings.simplefilter("ignore")
                        report = style_guide.check_files([tempfilename,])
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
        raise RuntimeError(
            "Code checker: invalid mode %i. Only 0, 1 or 2 supported" % mode
        )


if __name__ == "__main__":
    """small test run."""

    codestring = """def test(i : str , b) :
    a=2*3

    p = getCurrentPath()
    assert(1, 1)

    return a"""

    filename = "temp.py"
    fileSaved = False
    mode = 2  # all checks
    autoImportItom = True
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

    result = check(
        codestring, filename, fileSaved, mode, autoImportItom, furtherPropertiesJson,
    )

    import pprint

    pprint.pprint(result)
