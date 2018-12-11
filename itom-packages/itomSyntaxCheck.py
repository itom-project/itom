#load either pyflakes, or if not found frosted
try:
    from pyflakes import api
except ModuleNotFoundError:
    from frosted import api

#import re

class ItomReporter():
    """Formats the results of pyflakes / frosted checks and then presents them to the user."""
    def __init__(self):
        self.__unexpected_errors = []
        self.__flake = []
        self.__syntax_errors = []

    def unexpected_error(self, filename, msg):
        """
        An unexpected error occurred trying to process C{filename}.
        
        @param filename: The path to a file that we could not process.
        @ptype filename: C{unicode}
        @param msg: A message explaining the problem.
        @ptype msg: C{unicode}
        
        This method is called by frosted
        """
        return unexpectedError(filename, msg)
    
    def unexpectedError(self, filename, msg):
        """
        An unexpected error occurred trying to process C{filename}.
        
        @param filename: The path to a file that we could not process.
        @ptype filename: C{unicode}
        @param msg: A message explaining the problem.
        @ptype msg: C{unicode}
        
        This method is called by pyflakes
        """
        self.__unexpected_errors.append("%s: %s\n" % (filename, msg))
    
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
            self.__syntax_errors.append('%s:%d:%d: %s\n' %
                               (filename, lineno, offset + 1, msg))
        else:
            self.__syntax_errors.append('%s:%d: %s\n' % (filename, lineno, msg))

    def flake(self, message):
        """
        pyflakes found something wrong with the code.
        
        @param: A messages.Message.
        """
        self.__flake.append(str(message))
    
    def results(self):
        return ["\n".join(self.__unexpected_errors), "\n".join(self.__flake), "\n".join(self.__syntax_errors)]

def check(codestring):
    reporter = ItomReporter()
    api.check(codestring, "code", reporter = reporter)
    return reporter.results()