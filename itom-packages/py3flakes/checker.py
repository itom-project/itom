# -*- coding: utf-8 -*-

# Copyright (c) 2010 - 2012 Detlev Offenbach <detlev@die-offenbachs.de>
#
# Original (c) 2005-2010 Divmod, Inc.
#
# This module is based on pyflakes for Python2 but was heavily hacked to
# work with Python3 and eric5

import builtins
import os.path
import ast

from . import messages


class Binding(object):
    """
    Represents the binding of a value to a name.

    The checker uses this to keep track of which names have been bound and
    which names have not. See Assignment for a special type of binding that
    is checked with stricter rules.
    """
    def __init__(self, name, source):
        self.name = name
        self.source = source
        self.used = False

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<{0} object {1!r} from line {2!r} at 0x{3:x}>'.format(
            self.__class__.__name__,
            self.name,
            self.source.lineno,
            id(self))


class UnBinding(Binding):
    '''
    Created by the 'del' operator.
    '''


class Importation(Binding):
    """
    A binding created by an import statement.
    """
    def __init__(self, name, source):
        self.fullName = name
        name = name.split('.')[0]
        super(Importation, self).__init__(name, source)


class Argument(Binding):
    """
    Represents binding a name as an argument.
    """


class Assignment(Binding):
    """
    Represents binding a name with an explicit assignment.

    The checker will raise warnings for any Assignment that isn't used. Also,
    the checker does not consider assignments in tuple/list unpacking to be
    Assignments, rather it treats them as simple Bindings.
    """


class FunctionDefinition(Binding):
    """
    Represents a function definition.
    """
    pass


class ExportBinding(Binding):
    """
    A binding created by an __all__ assignment.  If the names in the list
    can be determined statically, they will be treated as names for export and
    additional checking applied to them.

    The only __all__ assignment that can be recognized is one which takes
    the value of a literal list containing literal strings.  For example::

        __all__ = ["foo", "bar"]

    Names which are imported and not otherwise used but appear in the value of
    __all__ will not have an unused import warning reported for them.
    """
    def names(self):
        """
        Return a list of the names referenced by this binding.
        """
        names = []
        if isinstance(self.source, ast.List):
            for node in self.source.elts:
                if isinstance(node, (ast.Str, ast.Bytes)):
                    names.append(node.s)
                elif isinstance(node, ast.Num):
                    names.append(node.n)
        return names


class Scope(dict):
    """
    Class defining the scope base class.
    """
    importStarred = False       # set to True when import * is found

    def __repr__(self):
        return '<{0} at 0x{1:x} {2}>'.format(
            self.__class__.__name__, id(self), dict.__repr__(self))

    def __init__(self):
        super(Scope, self).__init__()


class ClassScope(Scope):
    """
    Class representing a name scope for a class.
    """
    pass


class FunctionScope(Scope):
    """
    Class representing a name scope for a function.
    """
    def __init__(self):
        super(FunctionScope, self).__init__()
        self.globals = {}


class ModuleScope(Scope):
    """
    Class representing a name scope for a module.
    """
    pass

# Globally defined names which are not attributes of the builtins module.
_MAGIC_GLOBALS = ['__file__', '__builtins__']


class Checker(object):
    """
    Class to check the cleanliness and sanity of Python code.
    """
    nodeDepth = 0
    traceTree = False

    def __init__(self, module, filename='(none)'):
        """
        Constructor
        
        @param module parsed module tree or module source code
        @param filename name of the module file (string)
        """
        self._deferredFunctions = []
        self._deferredAssignments = []
        self.dead_scopes = []
        self.messages = []
        self.filename = filename
        self.scopeStack = [ModuleScope()]
        self.futuresAllowed = True
        
        if isinstance(module, str):
            module = ast.parse(module, filename, "exec")
        self.handleBody(module)
        self._runDeferred(self._deferredFunctions)
        # Set _deferredFunctions to None so that deferFunction will fail
        # noisily if called after we've run through the deferred functions.
        self._deferredFunctions = None
        self._runDeferred(self._deferredAssignments)
        # Set _deferredAssignments to None so that deferAssignment will fail
        # noisly if called after we've run through the deferred assignments.
        self._deferredAssignments = None
        del self.scopeStack[1:]
        self.popScope()
        self.check_dead_scopes()

    def deferFunction(self, callable):
        '''
        Schedule a function handler to be called just before completion.

        This is used for handling function bodies, which must be deferred
        because code later in the file might modify the global scope. When
        `callable` is called, the scope at the time this is called will be
        restored, however it will contain any new bindings added to it.
        '''
        self._deferredFunctions.append((callable, self.scopeStack[:]))

    def deferAssignment(self, callable):
        """
        Schedule an assignment handler to be called just after deferred
        function handlers.
        """
        self._deferredAssignments.append((callable, self.scopeStack[:]))

    def _runDeferred(self, deferred):
        """
        Run the callables in deferred using their associated scope stack.
        """
        for handler, scope in deferred:
            self.scopeStack = scope
            handler()

    def scope(self):
        return self.scopeStack[-1]
    scope = property(scope)

    def popScope(self):
        self.dead_scopes.append(self.scopeStack.pop())

    def check_dead_scopes(self):
        """
        Look at scopes which have been fully examined and report names in them
        which were imported but unused.
        """
        for scope in self.dead_scopes:
            export = isinstance(scope.get('__all__'), ExportBinding)
            if export:
                all = scope['__all__'].names()
                if os.path.split(self.filename)[1] != '__init__.py':
                    # Look for possible mistakes in the export list
                    undefined = set(all) - set(scope)
                    for name in undefined:
                        self.report(
                            messages.UndefinedExport,
                            scope['__all__'].source.lineno,
                            name)
            else:
                all = []

            # Look for imported names that aren't used.
            for importation in scope.values():
                if isinstance(importation, Importation):
                    if not importation.used and importation.name not in all:
                        self.report(
                            messages.UnusedImport,
                            importation.source.lineno,
                            importation.name)

    def pushFunctionScope(self):
        self.scopeStack.append(FunctionScope())

    def pushClassScope(self):
        self.scopeStack.append(ClassScope())

    def report(self, messageClass, *args, **kwargs):
        self.messages.append(messageClass(self.filename, *args, **kwargs))

    def handleBody(self, tree):
        for node in tree.body:
            self.handleNode(node, tree)

    def handleChildren(self, tree):
        for node in ast.iter_child_nodes(tree):
            self.handleNode(node, tree)
    
    def isDocstring(self, node):
        """
        Determine if the given node is a docstring, as long as it is at the
        correct place in the node tree.
        """
        return isinstance(node, ast.Str) or \
               (isinstance(node, ast.Expr) and
                isinstance(node.value, ast.Str))
    
    def handleNode(self, node, parent):
        if node:
            node.parent = parent
            if self.traceTree:
                print('  ' * self.nodeDepth + node.__class__.__name__)
            self.nodeDepth += 1
            if self.futuresAllowed and not \
                   (isinstance(node, ast.ImportFrom) or self.isDocstring(node)):
                self.futuresAllowed = False
            nodeType = node.__class__.__name__.upper()
            try:
                handler = getattr(self, nodeType)
                handler(node)
            finally:
                self.nodeDepth -= 1
            if self.traceTree:
                print('  ' * self.nodeDepth + 'end ' + node.__class__.__name__)

    def ignore(self, node):
        pass
    
    # ast nodes to be ignored
    PASS = CONTINUE = BREAK = ELLIPSIS = NUM = STR = BYTES = \
    LOAD = STORE = DEL = AUGLOAD = AUGSTORE = PARAM = \
    ATTRIBUTES = AND = OR = ADD = SUB = MULT = DIV = \
    MOD = POW = LSHIFT = RSHIFT = BITOR = BITXOR = BITAND = FLOORDIV = \
    INVERT = NOT = UADD = USUB = EQ = NOTEQ = LT = LTE = GT = GTE = IS = \
    ISNOT = IN = NOTIN = ignore

    # "stmt" type nodes
    RETURN = DELETE = PRINT = WHILE = IF = WITH = RAISE = TRYEXCEPT = \
        TRYFINALLY = ASSERT = EXEC = EXPR = handleChildren
    
    # "expr" type nodes
    BOOLOP = BINOP = UNARYOP = IFEXP = DICT = SET = YIELD = COMPARE = \
    CALL = REPR = ATTRIBUTE = SUBSCRIPT = LIST = TUPLE = handleChildren
    
    # "slice" type nodes
    SLICE = EXTSLICE = INDEX = handleChildren
    
    # additional node types
    COMPREHENSION = KEYWORD = handleChildren
    
    def addBinding(self, lineno, value, reportRedef=True):
        '''
        Called when a binding is altered.

        @param lineno line of the statement responsible for the change (integer)
        @param value the optional new value, a Binding instance, associated
            with the binding; if None, the binding is deleted if it exists
        @param reportRedef flag indicating if rebinding while unused will be
            reported (boolean)
        '''
        if (isinstance(self.scope.get(value.name), FunctionDefinition)
                    and isinstance(value, FunctionDefinition)):
            self.report(messages.RedefinedFunction,
                        lineno, value.name, self.scope[value.name].source.lineno)

        if not isinstance(self.scope, ClassScope):
            for scope in self.scopeStack[::-1]:
                existing = scope.get(value.name)
                if isinstance(existing, Importation) and \
                   not existing.used and \
                   not isinstance(value, UnBinding) and \
                   (not isinstance(value, Importation) or \
                    value.fullName == existing.fullName) and \
                   reportRedef:
                    self.report(messages.RedefinedWhileUnused,
                                lineno, value.name, scope[value.name].source.lineno)

        if isinstance(value, UnBinding):
            try:
                del self.scope[value.name]
            except KeyError:
                self.report(messages.UndefinedName, lineno, value.name)
        else:
            self.scope[value.name] = value
    
    ############################################################
    ## individual handler methods below
    ############################################################
    
    def GLOBAL(self, node):
        """
        Keep track of globals declarations.
        """
        if isinstance(self.scope, FunctionScope):
            self.scope.globals.update(dict.fromkeys(node.names))
    
    NONLOCAL = GLOBAL

    def LISTCOMP(self, node):
        for generator in node.generators:
            self.handleNode(generator, node)
        self.handleNode(node.elt, node)
    
    SETCOMP = GENERATOREXP = LISTCOMP
    
    def DICTCOMP(self, node):
        for generator in node.generators:
            self.handleNode(generator, node)
        self.handleNode(node.key, node)
        self.handleNode(node.value, node)
    
    def FOR(self, node):
        """
        Process bindings for loop variables.
        """
        vars = []

        def collectLoopVars(n):
            if isinstance(n, ast.Name):
                vars.append(n.id)
            elif isinstance(n, ast.expr_context):
                return
            else:
                for c in ast.iter_child_nodes(n):
                    collectLoopVars(c)

        collectLoopVars(node.target)
        for varn in vars:
            if (isinstance(self.scope.get(varn), Importation)
                    # unused ones will get an unused import warning
                    and self.scope[varn].used):
                self.report(messages.ImportShadowedByLoopVar,
                            node.lineno, varn, self.scope[varn].source.lineno)
        
        self.handleChildren(node)

    def NAME(self, node):
        """
        Handle occurrence of Name (which can be a load/store/delete access.)
        """
        # Locate the name in locals / function / globals scopes.
        if isinstance(node.ctx, (ast.Load, ast.AugLoad)):
            # try local scope
            importStarred = self.scope.importStarred
            try:
                self.scope[node.id].used = (self.scope, node.lineno)
            except KeyError:
                pass
            else:
                return

            # try enclosing function scopes
            for scope in self.scopeStack[-2:0:-1]:
                importStarred = importStarred or scope.importStarred
                if not isinstance(scope, FunctionScope):
                    continue
                try:
                    scope[node.id].used = (self.scope, node.lineno)
                except KeyError:
                    pass
                else:
                    return

            # try global scope
            importStarred = importStarred or self.scopeStack[0].importStarred
            try:
                self.scopeStack[0][node.id].used = (self.scope, node.lineno)
            except KeyError:
                if ((not hasattr(builtins, node.id))
                        and node.id not in _MAGIC_GLOBALS
                        and not importStarred):
                    if (os.path.basename(self.filename) == '__init__.py' and
                        node.id == '__path__'):
                        # the special name __path__ is valid only in packages
                        pass
                    else:
                        self.report(messages.UndefinedName, node.lineno, node.id)
        elif isinstance(node.ctx, (ast.Store, ast.AugStore)):
            # if the name hasn't already been defined in the current scope
            if isinstance(self.scope, FunctionScope) and node.id not in self.scope:
                # for each function or module scope above us
                for scope in self.scopeStack[:-1]:
                    if not isinstance(scope, (FunctionScope, ModuleScope)):
                        continue
                    # if the name was defined in that scope, and the name has
                    # been accessed already in the current scope, and hasn't
                    # been declared global
                    if (node.id in scope
                            and scope[node.id].used
                            and scope[node.id].used[0] is self.scope
                            and node.id not in self.scope.globals):
                        # then it's probably a mistake
                        self.report(messages.UndefinedLocal,
                                    scope[node.id].used[1],
                                    node.id,
                                    scope[node.id].source.lineno)
                        break

            if isinstance(node.parent,
                          (ast.For, ast.comprehension, ast.Tuple, ast.List)):
                binding = Binding(node.id, node)
            elif (node.id == '__all__' and
                  isinstance(self.scope, ModuleScope)):
                binding = ExportBinding(node.id, node.parent.value)
            else:
                binding = Assignment(node.id, node)
            if node.id in self.scope:
                binding.used = self.scope[node.id].used
            self.addBinding(node.lineno, binding)
        elif isinstance(node.ctx, ast.Del):
            if isinstance(self.scope, FunctionScope) and \
                   node.id in self.scope.globals:
                del self.scope.globals[node.id]
            else:
                self.addBinding(node.lineno, UnBinding(node.id, node))
        else:
            # must be a Param context -- this only happens for names in function
            # arguments, but these aren't dispatched through here
            raise RuntimeError(
                "Got impossible expression context: {0:r}".format(node.ctx,))

    def FUNCTIONDEF(self, node):
        if hasattr(node, "decorator_list"):
            for decorator in node.decorator_list:
                self.handleNode(decorator, node)
        self.addBinding(node.lineno, FunctionDefinition(node.name, node))
        self.LAMBDA(node)

    def LAMBDA(self, node):
        for default in node.args.defaults + node.args.kw_defaults:
            self.handleNode(default, node)

        def runFunction():
            args = []

            def addArgs(arglist):
                for arg in arglist:
                    if isinstance(arg.arg, tuple):
                        addArgs(arg.arg)
                    else:
                        if arg.arg in args:
                            self.report(messages.DuplicateArgument, node.lineno, arg.arg)
                        args.append(arg.arg)
            
            def checkUnusedAssignments():
                """
                Check to see if any assignments have not been used.
                """
                for name, binding in self.scope.items():
                    if (not binding.used and not name in self.scope.globals
                        and isinstance(binding, Assignment)):
                        self.report(messages.UnusedVariable,
                                    binding.source.lineno, name)

            self.pushFunctionScope()
            addArgs(node.args.args)
            addArgs(node.args.kwonlyargs)
            # vararg/kwarg identifiers are not Name nodes
            if node.args.vararg:
                args.append(node.args.vararg)
            if node.args.kwarg:
                args.append(node.args.kwarg)
            for name in args:
                self.addBinding(node.lineno, Argument(name, node), reportRedef=False)
            if isinstance(node.body, list):
                self.handleBody(node)
            else:
                self.handleNode(node.body, node)
            self.deferAssignment(checkUnusedAssignments)
            self.popScope()

        self.deferFunction(runFunction)

    def CLASSDEF(self, node):
        """
        Check names used in a class definition, including its decorators, base
        classes, and the body of its definition.  Additionally, add its name to
        the current scope.
        """
        for decorator in getattr(node, "decorator_list", []):
            self.handleNode(decorator, node)
        for baseNode in node.bases:
            self.handleNode(baseNode, node)
        self.addBinding(node.lineno, Binding(node.name, node))
        self.pushClassScope()
        self.handleBody(node)
        self.popScope()

    def handleAssignName(self, node):
        # special handling for ast.Subscript and ast.Starred
        if isinstance(node, (ast.Subscript, ast.Starred)):
            node.value.parent = node
            self.handleAssignName(node.value)
            if isinstance(node, ast.Subscript):
                if isinstance(node.slice, ast.Slice):
                    self.handleNode(node.slice.lower, node)
                    self.handleNode(node.slice.upper, node)
                else:
                    self.handleNode(node.slice.value, node)
            return
        
        # if the name hasn't already been defined in the current scope
        if isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                elt.parent = node
                self.handleAssignName(elt)
            return
        
        if isinstance(node, ast.Attribute):
            self.handleNode(node.value, node)
            return
        
        if isinstance(self.scope, FunctionScope) and node.id not in self.scope:
            # for each function or module scope above us
            for scope in self.scopeStack[:-1]:
                if not isinstance(scope, (FunctionScope, ModuleScope)):
                    continue
                # if the name was defined in that scope, and the name has
                # been accessed already in the current scope, and hasn't
                # been declared global
                if (node.id in scope
                        and scope[node.id].used
                        and scope[node.id].used[0] is self.scope
                        and node.id not in self.scope.globals):
                    # then it's probably a mistake
                    self.report(messages.UndefinedLocal,
                                scope[node.id].used[1],
                                node.id,
                                scope[node.id].source.lineno)
                    break

        if isinstance(node.parent,
                      (ast.For, ast.ListComp, ast.GeneratorExp,
                       ast.Tuple, ast.List)):
            binding = Binding(node.id, node)
        elif (node.id == '__all__' and
              isinstance(self.scope, ModuleScope) and
              isinstance(node.parent, ast.Assign)):
            binding = ExportBinding(node.id, node.parent.value)
        else:
            binding = Assignment(node.id, node)
        if node.id in self.scope:
            binding.used = self.scope[node.id].used
        self.addBinding(node.lineno, binding)

    def ASSIGN(self, node):
        self.handleNode(node.value, node)
        for target in node.targets:
            self.handleNode(target, node)
    
    def AUGASSIGN(self, node):
        # AugAssign is awkward: must set the context explicitly and visit twice,
        # once with AugLoad context, once with AugStore context
        node.target.ctx = ast.AugLoad()
        self.handleNode(node.target, node)
        self.handleNode(node.value, node)
        node.target.ctx = ast.AugStore()
        self.handleNode(node.target, node)
    
    def IMPORT(self, node):
        for alias in node.names:
            name = alias.asname or alias.name
            importation = Importation(name, node)
            self.addBinding(node.lineno, importation)

    def IMPORTFROM(self, node):
        if node.module == '__future__':
            if not self.futuresAllowed:
                self.report(messages.LateFutureImport, node.lineno,
                            [n.name for n in node.names])
        else:
            self.futuresAllowed = False

        for alias in node.names:
            if alias.name == '*':
                self.scope.importStarred = True
                self.report(messages.ImportStarUsed, node.lineno, node.module)
                continue
            name = alias.asname or alias.name
            importation = Importation(name, node)
            if node.module == '__future__':
                importation.used = (self.scope, node.lineno)
            self.addBinding(node.lineno, importation)
    
    def EXCEPTHANDLER(self, node):
        node.type and self.handleNode(node.type, node)
        if node.name:
            node.id = node.name
            self.handleAssignName(node)
        self.handleBody(node)
    
    def STARRED(self, node):
        self.handleNode(node.value, node)
