#load either pyflakes, or if not found frosted
try:
    from pyflakes import api
except ModuleNotFoundError:
    from frosted import api

class ItomReporter():
    """Formats the results of pyflakes / frosted checks and then presents them to the user."""
    def __init__(self):
        self.__unexpected_errors = []
        self.__flake = []

    def unexpected_error(self, filename, msg):
        """Output an unexpected_error specific to the provided filename."""
        self.__unexpected_errors.append("%s: %s\n" % (filename, msg))
        #self.stderr.write("%s: %s\n" % (filename, msg))

    def flake(self, message):
        """Print an error message to stdout."""
        self.__flake.append(str(message))
        #self.stdout.write(str(message))
        #self.stdout.write('\n')
    
    def results(self):
        return ["\n".join(self.__unexpected_errors), "\n".join(self.__flake)]

def check(codestring):
    reporter = ItomReporter()
    api.check(codestring, "code", reporter = reporter)
    return reporter.results()