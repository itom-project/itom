# ***********************************************************************
#    itom software
#    URL: http://www.uni-stuttgart.de/ito
#    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
#    Universitaet Stuttgart, Germany#
#
#    This file is part of itom.
#
#    itom is free software; you can redistribute it and/or modify it
#    under the terms of the GNU Library General Public Licence as published by
#    the Free Software Foundation; either version 2 of the Licence, or (at
#    your option) any later version.
#
#    itom is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
#    General Public Licence for more details.
#
#    You should have received a copy of the GNU Library General Public License
#    along with itom. If not, see <http://www.gnu.org/licenses/>.
# ***********************************************************************

import bdb
import itomDbgWrapper #given at runtime by itom-c++-project
import os
import re #regular expressions
import sys
import dis
import inspect
import traceback
import linecache

class Restart(Exception):
    """Causes a debugger to be restarted for the debugged python program."""
    pass

def find_function(funcname, filename):
    cre = re.compile(r'def\s+%s\s*[(]' % re.escape(funcname))
    try:
        fp = open(filename)
    except IOError:
        return None
    # consumer of this info expects the first line to be 1
    lineno = 1
    answer = None
    while True:
        line = fp.readline()
        if line == '':
            break
        if cre.match(line):
            answer = funcname, filename, lineno
            break
        lineno += 1
    fp.close()
    return answer

def getsourcelines(obj):
    lines, lineno = inspect.findsource(obj)
    if inspect.isframe(obj) and obj.f_globals is obj.f_locals:
        # must be a module frame: do not try to cut a block out of it
        return lines, 1
    elif inspect.ismodule(obj):
        return lines, 1
    return inspect.getblock(lines[lineno:]), lineno+1

def lasti2lineno(code, lasti):
    linestarts = list(dis.findlinestarts(code))
    linestarts.reverse()
    for i, lineno in linestarts:
        if lasti >= i:
            return lineno
    return 0


class _rstr(str):
    """String that doesn't quote its repr."""
    def __repr__(self):
        return self


class itoDebugger(bdb.Bdb):

    def __init__(self):
        bdb.Bdb.__init__(self, skip=None)
        self.mainpyfile = ''
        self._wait_for_mainpyfile = False
        self._wait_for_first_stop = True
        self._come_back_from_mainpyfile = False
        self.minStackIndexToDebug = 4
        self.debug = 0

        self.tb_lineno = {}

    def reset(self):
        bdb.Bdb.reset(self)
        self.forget()

    def forget(self):
        self.lineno = None
        self.stack = []
        self.curindex = 0
        self.curframe = None
        self.curframe_locals = None
        self.tb_lineno.clear()

    def setup(self, f, tb):
        self.forget()
        self.stack, self.curindex = self.get_stack(f, tb)
        while tb:
            # when setting up post-mortem debugging with a traceback, save all
            # the original line numbers to be displayed along the current line
            # numbers (which can be different, e.g. due to finally clauses)
            lineno = lasti2lineno(tb.tb_frame.f_code, tb.tb_lasti)
            self.tb_lineno[tb.tb_frame] = lineno
            tb = tb.tb_next
        self.curframe = self.stack[self.curindex][0]
        # The f_locals dictionary is updated from the actual frame
        # locals whenever the .f_locals accessor is called, so we
        # cache it here to ensure that modifications are not overwritten.
        self.curframe_locals = self.curframe.f_locals
        return

    def checkFrameForDebugging(self, frame):
        #checks the frame-stack to the bottom and verifies,
        #that the current stack-number is higher or equal to self.minStackIndexToDebug
        if(frame):
            count = 0
            temp_frame = frame
            while temp_frame:
                count = count +1
                temp_frame = temp_frame.f_back

            if count >= self.minStackIndexToDebug :
                return True
            else:
                return False

        else:
            return False


    # Override Bdb methods


    def user_call(self, frame, argument_list):
        """This method is called when there is the remote possibility
        that we ever need to stop in this function."""
        if(self.debug >= 2):
            print("user_call: I am in line %d of file %s" % (frame.f_lineno, frame.f_code.co_filename))
        if (self._wait_for_mainpyfile or not(self.checkFrameForDebugging(frame))):
            return
        if self.stop_here(frame):
            #self.message('--Call--')
            self.interaction(frame, None)


    def user_line(self, frame):
        """This function is called when we stop or break at this line."""
        #self.message("user line: at line: %d wait: %d" % (frame.f_lineno,  self._wait_for_mainpyfile))
        if(self.debug >= 1):
            print("user_line: I am in line %d of file %s" % (frame.f_lineno, frame.f_code.co_filename))
        #print("checkFrameForDebugging:", self.checkFrameForDebugging(frame))

        if not(self.checkFrameForDebugging(frame)):
            return

        if self._wait_for_mainpyfile:
            if (self.mainpyfile != self.canonic(frame.f_code.co_filename) or frame.f_lineno <= 0):
                return
            self._wait_for_mainpyfile = False
        #else:
            #print("user_line: I am in line %d of file %s, wait: %d" % (frame.f_lineno, frame.f_code.co_filename,  self._wait_for_mainpyfile))

        if(self.get_break(self.canonic(frame.f_code.co_filename),  frame.f_lineno)):
            #print("interaction due to breakpoint in line %d" % frame.f_lineno)
            self.interaction(frame, None)
        else:
            #print("interaction due to step in line %d" % frame.f_lineno)
            self.interaction(frame, None)




    def user_return(self, frame, return_value):
        """This function is called when a return trap is set here."""
        if(self.debug >= 2):
            print("user_return: I am in line %d of file %s" % (frame.f_lineno, frame.f_code.co_filename))
        #print("return")
        #print(frame)
        if self._wait_for_mainpyfile:
            return

        if(self.checkFrameForDebugging(frame)):
            #self._come_back_from_mainpyfile = True
            frame.f_locals['__return__'] = return_value
            #print('--Return (End)--')
            #self.interaction(frame, None)
        else:
            frame.f_locals['__return__'] = return_value
            #print('--Return--')


    def user_exception(self, frame, exc_info):
        """This function is called if an exception occurs,
        but only if we are to stop at or just below this level."""
        if self._wait_for_mainpyfile:
            return

        #in case of an exception, you can directly stop...
        #return
        #... or you can print the traceback while passing the exception through the whole traceback-stack

        exc_type, exc_value, exc_traceback = exc_info
        frame.f_locals['__exception__'] = exc_type, exc_value, exc_traceback

        #if issubclass(exc_type, SyntaxError):
            #this is a hack, since exc_value is no instance of an error, but only a tuple with the arguments
            #However, in case of a SyntaxError, format_exception_only below needs an instance of SyntaxError
            #exc_value = SyntaxError(exc_value[0], exc_value[1])

        if(not exc_traceback is None):
            print("exception raised in method '" + exc_traceback.tb_frame.f_code.co_name + "', line " + str(exc_traceback.tb_frame.f_lineno))
        print(traceback.format_exception_only(exc_type, exc_value)[-1].strip())
        self.interaction(frame, exc_traceback)

    # General interaction function
    def _cmdloop(self,  frame):
        if(self._wait_for_first_stop == False):
            itomDbgWrapper.pyDbgCommandLoop(self, frame)
        elif(self.get_break(self.canonic(frame.f_code.co_filename),  frame.f_lineno)):
            self._wait_for_first_stop = False
            #this is the case that the first call of _cmdloop should usually be ignored, but it is a breakpoint-line. Therefore stop.
            itomDbgWrapper.pyDbgCommandLoop(self, frame)
        else:
            self._wait_for_first_stop = False
            self.set_continue()


    def interaction(self, frame, traceback):
        self.setup(frame, traceback) #save frame parameters
        #self.print_stack_entry(self.stack[self.curindex])
        self._cmdloop(frame)
        self.forget() #delete frame parameters



    # interface abstraction functions

    def message(self, msg):
        print(msg)

    def error(self, msg):
        print('***', msg)

    # Command definitions, called by cmdloop()
    # The argument is the remaining string on the command line
    # Return true to exit from the command loop

    # To be overridden in derived debuggers
    def defaultFile(self):
        """Produce a reasonable default."""
        filename = self.curframe.f_code.co_filename
        if filename == '<string>' and self.mainpyfile:
            filename = self.mainpyfile
        return filename

    def addNewBreakPoint(self,  filename,  lineno,  enabled,  temporary,  condition,  ignoreCount):
        """Adds breakpoint to list of breakpoints.

        Return breakpoint ID (int) or error string (str).

        This method should not raise an exception.
        """
        if not filename:
            filename = self.defaultFile()
        else:
            filename = self.canonic(filename)

        # Check for reasonable breakpoint
        try:
            line = self._checkline(filename, lineno) #may raise an UnicodeDecodeError or IndexError
        except IndexError as err:
            return "Cannot add breakpoint: " + str(err)
        except UnicodeDecodeError as err:
            # The _ in the error string indicates to not delete the break point.
            return "_Cannot add breakpoint: " + str(err)

        if line > 0:
            # now set the break point
            err = self.set_break(filename, line, temporary, condition, None) #returns None if ok, else an error string
            if err is None:
                bp = self.get_breaks(filename, line)[-1]
                bp.enabled = enabled
                bp.ignore = ignoreCount
                return int(bp.number)
            else:
                return "Cannot add breakpoint (file '%s', line %i): " % (filename, lineno) + str(err)

    def editBreakPoint(self,  bpNumber,  filename,  lineno,  enabled,  temporary,  condition,  ignoreCount):
        """
        Edit breakpoint given by bpNumber.

        Return 0 if successful, else str with error information
        """
        if not filename:
            filename = self.defaultFile()
        else:
            filename = self.canonic(filename) # This better be in canonical form!

        # Check for reasonable breakpoint
        try:
            line = self._checkline(filename, lineno) #may raise an UnicodeDecodeError or IndexError
        except IndexError as err:
            return "Cannot edit breakpoint: " + str(err)
        except UnicodeDecodeError as err:
            # The _ in the error string indicates to not delete the break point.
            return "_Cannot edit breakpoint: " + str(err)

        if line > 0:
            #input values are ok, break point can be edited
            try:
                bp = self.get_bpbynumber(bpNumber)
            except ValueError as err:
                return "Cannot edit breakpoint: " + str(err)
            else:
                bp.file = filename
                bp.line = line
                bp.temporary = temporary
                bp.cond = condition
                bp.enabled = enabled
                bp.ignore = ignoreCount
                return 0
        else:
            return "Cannot edit breakpoint: Line %i in file \"%s\" is an invalid line number, a blank or comment line"  % (lineno, filename)

    def clearBreakPoint(self, bpNumber):
        """Clears breakpoint with given bpNumber.

        return 0 if done, else str with error
        """
        try:
            self.get_bpbynumber(bpNumber) #check if bp-number exists, if not raises a ValueError
        except ValueError as err:
            return "Cannot clear breakpoint: " + str(err)
        self.clear_bpbynumber(bpNumber)
        return 0

    def _checkline(self, filename, lineno):
        """Check whether specified line seems to be executable.

        Return `lineno` if it is or raise an IndexError if not (e.g. a docstring, comment, blank
        line or EOF). Warning: testing is not comprehensive.
        """
        # this method should be callable before starting debugging, so default
        # to "no globals" if there is no current frame
        if hasattr(self, 'curframe') and self.curframe != None:
            globs = self.curframe.f_globals
        else:
            globs = None

        linecache.checkcache(filename) #force linecache to check if file has been update since last getline
        #globs = self.curframe.f_globals if hasattr(self, 'curframe') else None
        try:
            line = linecache.getline(filename, lineno, globs) #returns line in file or '' if empty line or invalid file or lineno
        except UnicodeDecodeError as E:
            E.reason = "{0}. File '{1}', line {2}".format(E.reason,filename,lineno)
            raise #reraise modified error

        if(line == ''):
            raise IndexError("Line %d in file \"%s\" is blank or does not exist." % (lineno, filename))
        else:
            line = line.strip()
            # Don't allow setting breakpoint at a blank line
            if (not line or (line[0] == '#') or (line[:3] == '"""') or line[:3] == "'''"):
                raise IndexError("Line %d in file \"%s\" is blank or is a comment." % (lineno, filename))

        return lineno

    def do_where(self, arg):
        """w(here)
        Print a stack trace, with the most recent frame at the bottom.
        An arrow indicates the "current frame", which determines the
        context of most commands.  'bt' is an alias for this command.
        """
        self.print_stack_trace()

    def do_clear(self, arg):
        """Remove temporary breakpoint.

        Must implement in derived classes or get NotImplementedError.
        """
        try:
            bpNumber = int(arg)
        except Exception:
            return

        itomDbgWrapper.pyDbgClearBreakpoint(bpNumber)

    def _select_frame(self, number):
        assert 0 <= number < len(self.stack)
        self.curindex = number
        self.curframe = self.stack[self.curindex][0]
        self.curframe_locals = self.curframe.f_locals
        self.print_stack_entry(self.stack[self.curindex])
        self.lineno = None

    def do_quit(self, arg):
        """q(uit)\nexit
        Quit from the debugger. The program being executed is aborted.
        """
        self._user_requested_quit = True
        self.set_quit()
        return 1

    def do_args(self, arg):
        """a(rgs)
        Print the argument list of the current function.
        """
        co = self.curframe.f_code
        dict = self.curframe_locals
        n = co.co_argcount
        if co.co_flags & 4: n = n+1
        if co.co_flags & 8: n = n+1
        for i in range(n):
            name = co.co_varnames[i]
            if name in dict:
                self.message('%s = %r' % (name, dict[name]))
            else:
                self.message('%s = *** undefined ***' % (name,))

    def do_retval(self, arg):
        """retval
        Print the return value for the last return of a function.
        """
        if '__return__' in self.curframe_locals:
            self.message(repr(self.curframe_locals['__return__']))
        else:
            self.error('Not yet returned!')

    def _getval(self, arg):
        try:
            return eval(arg, self.curframe.f_globals, self.curframe_locals)
        except Exception:
            exc_info = sys.exc_info()[:2]
            self.error(traceback.format_exception_only(*exc_info)[-1].strip())
            raise

    def _getval_except(self, arg, frame=None):
        try:
            if frame is None:
                return eval(arg, self.curframe.f_globals, self.curframe_locals)
            else:
                return eval(arg, frame.f_globals, frame.f_locals)
        except Exception:
            exc_info = sys.exc_info()[:2]
            err = traceback.format_exception_only(*exc_info)[-1].strip()
            return _rstr('** raised %s **' % err)

    def do_p(self, arg):
        """p(rint) expression
        Print the value of the expression.
        """
        try:
            self.message(repr(self._getval(arg)))
        except Exception:
            pass
    # make "print" an alias of "p" since print isn't a Python statement anymore

    def do_pp(self, arg):
        """pp expression
        Pretty-print the value of the expression.
        """
        try:
            self.message(pprint.pformat(self._getval(arg)))
        except Exception:
            pass

    def do_list(self, arg):
        """l(ist) [first [,last] | .]

        List source code for the current file.  Without arguments,
        list 11 lines around the current line or continue the previous
        listing.  With . as argument, list 11 lines around the current
        line.  With one argument, list 11 lines starting at that line.
        With two arguments, list the given range; if the second
        argument is less than the first, it is a count.

        The current line in the current frame is indicated by "->".
        If an exception is being debugged, the line where the
        exception was originally raised or propagated is indicated by
        ">>", if it differs from the current line.
        """
        self.lastcmd = 'list'
        last = None
        if arg and arg != '.':
            try:
                if ',' in arg:
                    first, last = arg.split(',')
                    first = int(first.strip())
                    last = int(last.strip())
                    if last < first:
                        # assume it's a count
                        last = first + last
                else:
                    first = int(arg.strip())
                    first = max(1, first - 5)
            except ValueError:
                self.error('Error in argument: %r' % arg)
                return
        elif self.lineno is None or arg == '.':
            first = max(1, self.curframe.f_lineno - 5)
        else:
            first = self.lineno + 1
        if last is None:
            last = first + 10
        filename = self.curframe.f_code.co_filename
        breaklist = self.get_file_breaks(filename)
        try:
            lines = linecache.getlines(filename, self.curframe.f_globals)
            self._print_lines(lines[first-1:last], first, breaklist,
                              self.curframe)
            self.lineno = min(last, len(lines))
            if len(lines) < last:
                self.message('[EOF]')
        except KeyboardInterrupt:
            pass

    def do_longlist(self, arg):
        """longlist | ll
        List the whole source code for the current function or frame.
        """
        filename = self.curframe.f_code.co_filename
        breaklist = self.get_file_breaks(filename)
        try:
            lines, lineno = getsourcelines(self.curframe)
        except IOError as err:
            self.error(err)
            return
        self._print_lines(lines, lineno, breaklist, self.curframe)

    def do_source(self, arg):
        """source expression
        Try to get source code for the given object and display it.
        """
        try:
            obj = self._getval(arg)
        except Exception:
            return
        try:
            lines, lineno = getsourcelines(obj)
        except (IOError, TypeError) as err:
            self.error(err)
            return
        self._print_lines(lines, lineno)

    def _print_lines(self, lines, start, breaks=(), frame=None):
        """Print a range of lines."""
        if frame:
            current_lineno = frame.f_lineno
            exc_lineno = self.tb_lineno.get(frame, -1)
        else:
            current_lineno = exc_lineno = -1
        for lineno, line in enumerate(lines, start):
            s = str(lineno).rjust(3)
            if len(s) < 4:
                s += ' '
            if lineno in breaks:
                s += 'B'
            else:
                s += ' '
            if lineno == current_lineno:
                s += '->'
            elif lineno == exc_lineno:
                s += '>>'
            self.message(s + '\t' + line.rstrip())

    def do_whatis(self, arg):
        """whatis arg
        Print the type of the argument.
        """
        try:
            value = self._getval(arg)
        except Exception:
            # _getval() already printed the error
            return
        code = None
        # Is it a function?
        try:
            code = value.__code__
        except Exception:
            pass
        if code:
            self.message('Function %s' % code.co_name)
            return
        # Is it an instance method?
        try:
            code = value.__func__.__code__
        except Exception:
            pass
        if code:
            self.message('Method %s' % code.co_name)
            return
        # Is it a class?
        if value.__class__ is type:
            self.message('Class %s.%s' % (value.__module__, value.__name__))
            return
        # None of the above...
        self.message(type(value))





    # Print a traceback starting at the top stack frame.
    # The most recently entered frame is printed last;
    # this is different from dbx and gdb, but consistent with
    # the Python interpreter's stack trace.
    # It is also consistent with the up/down commands (which are
    # compatible with dbx and gdb: up moves towards 'main()'
    # and down moves towards the most recent stack frame).

    def print_stack_trace(self):
        for frame_lineno in self.stack:
            self.print_stack_entry(frame_lineno)


    def print_stack_entry(self, frame_lineno, prompt_prefix=':::'):
        frame, lineno = frame_lineno
        if frame is self.curframe:
            prefix = '> '
        else:
            prefix = '  '
        self.message(prefix +
                     self.format_stack_entry(frame_lineno, prompt_prefix))



    # other helper functions

    def lookupmodule(self, filename):
        """Helper function for break/clear parsing -- may be overridden.

        lookupmodule() translates (possibly incomplete) file or module name
        into an absolute file name.
        """
        if os.path.isabs(filename) and  os.path.exists(filename):
            return filename
        f = os.path.join(sys.path[0], filename)
        if  os.path.exists(f) and self.canonic(f) == self.mainpyfile:
            return f
        root, ext = os.path.splitext(filename)
        if ext == '':
            filename = filename + '.py'
        if os.path.isabs(filename):
            return filename
        for dirname in sys.path:
            while os.path.islink(dirname):
                dirname = os.readlink(dirname)
            fullname = os.path.join(dirname, filename)
            if os.path.exists(fullname):
                return fullname
        return None


    def checkSysPath(self,canonicFilename):
        [dir,file] = os.path.split(canonicFilename)
        if( (dir in sys.path) == False):
            sys.path.append(dir)

    def debugString(self, codeString):
        # The script has to run in __main__ namespace (or imports from
        # __main__ will break).
        import __main__
        self.minStackIndexToDebug = 4

        tempFilename = ""
        if(hasattr(__main__.__dict__ ,  "__file__")):
            tempFilename = __main__.__dict__.__file__

        __main__.__dict__.update({"__file__" : "<string>"})

        self._wait_for_mainpyfile = False #must be false, otherwise the debugger will not start since there is no function-stack by compiler and exec command, which forces us to check when the real function is finally reached.
        self._wait_for_first_stop = True #if True we are waiting for the first stop (first line), where the debugger is forced to directly continue, if False the debugger is forced to also stop in the first possible line
        self._come_back_from_mainpyfile = False
        self.mainpyfile = "<string>"
        self._user_requested_quit = False

        self.reset()
        try:
            compiledCode = compile(codeString, "<string>", mode = "single") #mode = 'single' forces statements that evaluate to something other than None will be printed
            self.runeval(compiledCode)
        finally:
            self.clear_all_breaks()
            self.reset()

            if(tempFilename != ""):
                __main__.__dict__.update({"__file__" : tempFilename})
            else:
                del(__main__.__dict__["__file__"])

    def debugFunction(self, fctPointer, args, kwargs = None):
        # The script has to run in __main__ namespace (or imports from
        # __main__ will break).
        import __main__
        self.minStackIndexToDebug = 3

        tempFilename = ""
        if(hasattr(__main__.__dict__ ,  "__file__")):
            tempFilename = __main__.__dict__.__file__

        __main__.__dict__.update({"__file__" : "<string>"})

        self._wait_for_mainpyfile = False #must be false, otherwise the debugger will not start since there is no function-stack by compiler and exec command, which forces us to check when the real function is finally reached.
        self._wait_for_first_stop = True #if True we are waiting for the first stop (first line), where the debugger is forced to directly continue, if False the debugger is forced to also stop in the first possible line
        self._come_back_from_mainpyfile = False
        self.mainpyfile = "<string>"
        self._user_requested_quit = False

        self.reset()
        try:
            #for arg in args:
            #    print ("another arg:", arg)
            #for key in kwargs:
            #    print ("another keyword arg: %s: %s" % (key, kwargs[key]))
            if(kwargs is None):
                self.runcall(fctPointer,*args)
            else:
                self.runcall(fctPointer,*args,**kwargs)
            self.clear_all_breaks()
            self.reset()
        finally:
            if(tempFilename != ""):
                __main__.__dict__.update({"__file__" : tempFilename})
            else:
                del(__main__.__dict__["__file__"])

    def parseUnicodeError(self, err):
        import re
        m = re.match(r"\(unicode error\) (.*) can't decode byte (0x[0-9a-zA-Z]{1,2}) in position (\d+): (.*)", err.msg)
        if m:
            if err.filename is None or err.filename == "" or err.lineno <= 0:
                err.msg = "(unicode error) %s cannot decode byte '%s' in line %i, position %i: %s. \nThe line possibly contains an invalid character. Please remove them or add a coding hint in the first line (menu option 'insert codec...')" \
                    % (m.group(1), m.group(2), err.lineno, int(m.group(3)), m.group(4))
            else:
                wrongByte = int(m.group(2), 16)
                fp = open(err.filename, "rb")
                text = fp.read()
                fp.close()
                lines = text.split(b"\n")
                if len(lines) >= err.lineno:
                    err.msg = "(unicode error) %s cannot decode byte '%s' in line %i, position %i: %s. \nThe line possibly contains an invalid character. Please remove them or add a coding hint in the first line (menu option 'insert codec...')" \
                        % (m.group(1), m.group(2), err.lineno, lines[err.lineno - 1].index(wrongByte) + 1, m.group(4))
                else:
                    err.msg = "(unicode error) %s cannot decode byte '%s' in line %i, position %i: %s. \nThe line possibly contains an invalid character. Please remove them or add a coding hint in the first line (menu option 'insert codec...')" \
                        % (m.group(1), m.group(2), err.lineno, int(m.group(3)), m.group(4))
            raise #reraise exception
        else:
            raise #reraise exception

    def debugScript(self, filename):
        # The script has to run in __main__ namespace (or imports from
        # __main__ will break).
        #
        import __main__
        self.minStackIndexToDebug = 4

        tempFilename = ""
        if(hasattr(__main__.__dict__ ,  "__file__")):
            tempFilename = __main__.__dict__.__file__

        __main__.__dict__.update({"__file__" : filename})

        self._wait_for_mainpyfile = True
        self._wait_for_first_stop = False #if True we are waiting for the first stop (first line), where the debugger is forced to directly continue
        self._come_back_from_mainpyfile = False
        self.mainpyfile = self.canonic(filename)
        self._user_requested_quit = False

        self.checkSysPath(self.mainpyfile)

        self.reset()

        with open(filename, "rb") as fp:
            statement = "exec(compile(%r,%r,'exec'))" % \
                        (fp.read(), self.mainpyfile)
        try:
            self.run(statement,  __main__.__dict__)
            self.clear_all_breaks()
            self.reset()
        except SyntaxError as err:
            if err.msg.startswith('(unicode error)'):
                self.parseUnicodeError(err)
            else:
                raise
        finally:
            if(tempFilename != ""):
                __main__.__dict__.update({"__file__" : tempFilename})
            else:
                del(__main__.__dict__["__file__"])


    def runScript(self,  filename):

        import __main__
        self.minStackIndexToDebug = 4

        tempFilename = ""
        if(hasattr(__main__.__dict__ ,  "__file__")):
            tempFilename = __main__.__dict__.__file__

        __main__.__dict__.update({"__file__" : filename})

        self.mainpyfile = self.canonic(filename)
        self.checkSysPath(self.mainpyfile)

        with open(filename, "rb") as fp:
            statement = "exec(compile(%r,%r,'exec'))" % \
                        (fp.read(), self.mainpyfile)
        try:
            exec(statement,  __main__.__dict__)
        except SyntaxError as err:
            if err.msg.startswith('(unicode error)'):
                self.parseUnicodeError(err)
            else:
                raise
        finally:
            if(tempFilename != ""):
                __main__.__dict__.update({"__file__" : tempFilename})
            else:
                del(__main__.__dict__["__file__"])

    def compileScript(self,  filename):

        import __main__
        self.minStackIndexToDebug = 4

        tempFilename = ""
        if(hasattr(__main__.__dict__ ,  "__file__")):
            tempFilename = __main__.__dict__.__file__

        __main__.__dict__.update({"__file__" : filename})

        with open(filename, "rb") as fp:
            statement = "compile(%r,%r,'exec')" % \
                        (fp.read(), self.mainpyfile)
        exec(statement,  __main__.__dict__)

        if(tempFilename != ""):
            __main__.__dict__.update({"__file__" : tempFilename})
        else:
            del(__main__.__dict__["__file__"])
