#load either pyflakes, or if not found frosted
from pyflakes import api as pyflakesapi
from typing import Optional
import tempfile
import os

try:
    from flake8.api import legacy as flake8
    HAS_FLAKE8 = True
except:
    HAS_FLAKE8 = False

class ItomFlakesReporter():
    """Formats the results of pyflakes checks and then presents them to the user."""
    def __init__(self, filename : str):
        self.__items = []
        self.__filename = filename
    
    def __addItem__(self, type : int, filename : str, msgCode : str, description : str, lineNo : int = -1, column : int = -1):
        '''
        @param type: the type of message (0: Info, 1: Warning, 2: Error)
        @ptype type: C{int}
        '''
        self.__items.append("%i::%s::%i::%i::%s::%s" % (type, self.__filename, lineNo, column, msgCode, description))
        
    
    def unexpectedError(self, filename, msg):
        """
        An unexpected error occurred trying to process C{filename}.
        
        @param filename: The path to a file that we could not process.
        @ptype filename: C{unicode}
        @param msg: A message explaining the problem.
        @ptype msg: C{unicode}
        
        This method is called by pyflakes
        """
        self.__addItem__(type = 2, filename = filename, msgCode = "", description = msg, lineNo = -1, column = -1)
    
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
            self.__addItem__(type = 2, filename = filename, msgCode = "", 
                            description = msg, lineNo = lineno, column = offset + 1)
        else:
            self.__addItem__(type = 2, filename = filename, msgCode = "", 
                            description = msg, lineNo = lineno, column = -1)

    def flake(self, message):
        """
        pyflakes found something wrong with the code.
        
        @param: A messages.Message.
        """
        msg = message.message % message.message_args
        self.__addItem__(type = 2, filename = message.filename, msgCode = "", 
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
        return self.__items

def check(codestring : str, filename : str, fileSaved : bool) -> str:
    '''run the test for a single file.
    '''
    print(filename, fileSaved)
    if HAS_FLAKE8:
        style_guide = flake8.get_style_guide(ignore=[])
        
        if fileSaved:
            report = style_guide.input_file(filename)
        else:
            with tempfile.NamedTemporaryFile("wt", delete = False, suffix = ".py") as fp:
                tempfilename = fp.name
                print("temp filename", tempfilename)
                fp.write(codestring)
            
            try:
                report = style_guide.input_file(tempfilename)
            except:
                pass
            finally:
                os.remove(tempfilename)
        
        
        print("Run flake8 on file %s: %i" % (filename, report.total_errors))
        #pass
    
    reporter = ItomFlakesReporter(filename)
    pyflakesapi.check(codestring, "code", reporter = reporter)
    return reporter.results()