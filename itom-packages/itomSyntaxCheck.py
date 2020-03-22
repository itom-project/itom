#load either pyflakes, or if not found frosted
from pyflakes import api as pyflakesapi
from typing import Optional
import tempfile
import os

try:
    from flake8.api import legacy as flake8
    HAS_FLAKE8 = True
    from flake8.formatting import base
    
    if False:  # `typing.TYPE_CHECKING` was introduced in 3.5.2
        from flake8.style_guide import Violation
except:
    HAS_FLAKE8 = False

###############################################################
class ItomFlakesReporter():
    """Formats the results of pyflakes checks and then presents them to the user."""
    def __init__(self, filename : str):
        self._items = []
        self._filename = filename
    
    def _addItem(self, type : int, filename : str, msgCode : str, description : str, lineNo : int = -1, column : int = -1):
        '''
        @param type: the type of message (0: Info, 1: Warning, 2: Error)
        @ptype type: C{int}
        '''
        self._items.append("%i::%s::%i::%i::%s::%s" % (type, self._filename, lineNo, column, msgCode, description))
        
    
    def unexpectedError(self, filename, msg):
        """
        An unexpected error occurred trying to process C{filename}.
        
        @param filename: The path to a file that we could not process.
        @ptype filename: C{unicode}
        @param msg: A message explaining the problem.
        @ptype msg: C{unicode}
        
        This method is called by pyflakes
        """
        self._addItem(type = 2, filename = filename, msgCode = "", description = msg, lineNo = -1, column = -1)
    
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
            self._addItem(type = 2, filename = filename, msgCode = "", 
                            description = msg, lineNo = lineno, column = offset + 1)
        else:
            self._addItem(type = 2, filename = filename, msgCode = "", 
                            description = msg, lineNo = lineno, column = -1)

    def flake(self, message):
        """
        pyflakes found something wrong with the code.
        
        @param: A messages.Message.
        """
        msg = message.message % message.message_args
        self._addItem(type = 2, filename = message.filename, msgCode = "", 
                    description = msg, lineNo = message.lineno, column = message.col)
    
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
    
    def _addItem(self, errorType : int, filename : str, msgCode : str, description : str, lineNo : int = -1, column : int = -1):
        '''
        @param type: the type of message (0: Info, 1: Warning, 2: Error)
        @ptype type: C{int}
        '''
        self._items.append("%i::%s::%i::%i::%s::%s" % (errorType, filename, lineNo, column, msgCode, description))
    
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
        if error.code.startswith("F"):
            errorType = 2 #error
        else:
            errorType = 1 #warning
        
        self._addItem(errorType, error.filename, error.code, error.text, error.line_number, error.column_number)

#############################################################
def check(codestring : str, filename : str, fileSaved : bool) -> str:
    '''run the test for a single file.
    '''
    #print(filename, fileSaved)
    if HAS_FLAKE8:
        style_guide = flake8.get_style_guide()
        style_guide.init_report(reporter = ItomFlake8Formatter)
        report = None
        
        if fileSaved:
            report = style_guide.check_files([filename,])
        else:
            with tempfile.NamedTemporaryFile("wt", delete = False, suffix = ".py") as fp:
                tempfilename = fp.name
                fp.write(codestring)
            
            try:
                report = style_guide.check_files([tempfilename,])
                
            except:
                pass
            finally:
                os.remove(tempfilename)
        
        if report is not None:
            results = report._application.formatter.results()
            #print("Run flake8 on file %s: %i" % (filename, report.total_errors))
            return results
        else:
            return []
    else:
        reporter = ItomFlakesReporter(filename)
        pyflakesapi.check(codestring, "code", reporter = reporter)
        return reporter.results()