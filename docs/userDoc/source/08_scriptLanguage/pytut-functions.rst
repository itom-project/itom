﻿

.. include:: ../include/global.inc

Functions
==========
.. moduleauthor:: PSchau
.. sectionauthor:: PSchau




Defining Functions
-------------------

We can create a function that writes the Fibonacci series to an arbitrary
boundary::

   >>> def fib(n):    # write Fibonacci series up to n
   ...     """Print a Fibonacci series up to n."""
   ...     a, b = 0, 1
   ...     while a < n:
   ...         print(a, end=' ')
   ...         a, b = b, a+b
   ...     print()
   ...
   >>> # Now call the function we just defined:
   ... fib(2000)
   0 1 1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597

.. index::
   single: documentation strings
   single: docstrings
   single: strings, documentation

The keyword `def` introduces a function *definition*.  It must be
followed by the function name and the parenthesized list of formal parameters.
The statements that form the body of the function start at the next line, and
must be indented.

The first statement of the function body can optionally be a string literal;
this string literal is the function's documentation string, or :dfn:`docstring`.
(More about docstrings can be found in the section :ref:`tut-docstrings`.)
There are tools which use docstrings to automatically produce online or printed
documentation, or to let the user interactively browse through code; it's good
practice to include docstrings in code that you write, so make a habit of it.

The *execution* of a function introduces a new symbol table used for the local
variables of the function.  More precisely, all variable assignments in a
function store the value in the local symbol table; whereas variable references
first look in the local symbol table, then in the local symbol tables of
enclosing functions, then in the global symbol table, and finally in the table
of built-in names. Thus, global variables cannot be directly assigned a value
within a function (unless named in a `global` statement), although they
may be referenced.

The actual parameters (arguments) to a function call are introduced in the local
symbol table of the called function when it is called; thus, arguments are
passed using *call by value* (where the *value* is always an object *reference*,
not the value of the object). [#]_ When a function calls another function, a new
local symbol table is created for that call.

A function definition introduces the function name in the current symbol table.
The value of the function name has a type that is recognized by the interpreter
as a user-defined function.  This value can be assigned to another name which
can then also be used as a function.  This serves as a general renaming
mechanism::

   >>> fib
   <function fib at 10042ed0>
   >>> f = fib
   >>> f(100)
   0 1 1 2 3 5 8 13 21 34 55 89

Coming from other languages, you might object that ``fib`` is not a function but
a procedure since it doesn't return a value.  In fact, even functions without a
`return` statement do return a value, albeit a rather boring one.  This
value is called ``None`` (it's a built-in name).  Writing the value ``None`` is
normally suppressed by the interpreter if it would be the only value written.
You can see it if you really want to using :func:`print`::

   >>> fib(0)
   >>> print(fib(0))
   None

It is simple to write a function that returns a list of the numbers of the
Fibonacci series, instead of printing it::

   >>> def fib2(n): # return Fibonacci series up to n
   ...     """Return a list containing the Fibonacci series up to n."""
   ...     result = []
   ...     a, b = 0, 1
   ...     while a < n:
   ...         result.append(a)    # see below
   ...         a, b = b, a+b
   ...     return result
   ...
   >>> f100 = fib2(100)    # call it
   >>> f100                # write the result
   [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

This example, as usual, demonstrates some new Python features:

* The `return` statement returns with a value from a function.
  `return` without an expression argument returns ``None``. Falling off
  the end of a function also returns ``None``.

* The statement ``result.append(a)`` calls a *method* of the list object
  ``result``.  A method is a function that 'belongs' to an object and is named
  ``obj.methodname``, where ``obj`` is some object (this may be an expression),
  and ``methodname`` is the name of a method that is defined by the object's type.
  Different types define different methods.  Methods of different types may have
  the same name without causing ambiguity.  (It is possible to define your own
  object types and methods, using *classes*, see :ref:`tut-classes`)
  The method :meth:`append` shown in the example is defined for list objects; it
  adds a new element at the end of the list.  In this example it is equivalent to
  ``result = result + [a]``, but more efficient.
  
  
More on Defining Functions
---------------------------

It is also possible to define functions with a variable number of arguments.
There are three forms, which can be combined.


.. _tut-defaultargs:

Default Argument Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most useful form is to specify a default value for one or more arguments.
This creates a function that can be called with fewer arguments than it is
defined to allow.  For example::

   def ask_ok(prompt, retries=4, complaint='Yes or no, please!'):
       while True:
           ok = input(prompt)
           if ok in ('y', 'ye', 'yes'):
               return True
           if ok in ('n', 'no', 'nop', 'nope'):
               return False
           retries = retries - 1
           if retries < 0:
               raise IOError('refusenik user')
           print(complaint)

This function can be called in several ways:

* giving only the mandatory argument:
  ``ask_ok('Do you really want to quit?')``
* giving one of the optional arguments:
  ``ask_ok('OK to overwrite the file?', 2)``
* or even giving all arguments:
  ``ask_ok('OK to overwrite the file?', 2, 'Come on, only yes or no!')``

This example also introduces the :keyword:`in` keyword. This tests whether or
not a sequence contains a certain value.

The default values are evaluated at the point of function definition in the
*defining* scope, so that ::

   i = 5

   def f(arg=i):
       print(arg)

   i = 6
   f()

will print ``5``.

**Important warning:**  The default value is evaluated only once. This makes a
difference when the default is a mutable object such as a list, dictionary, or
instances of most classes.  For example, the following function accumulates the
arguments passed to it on subsequent calls::

   def f(a, L=[]):
       L.append(a)
       return L

   print(f(1))
   print(f(2))
   print(f(3))

This will print ::

   [1]
   [1, 2]
   [1, 2, 3]

If you don't want the default to be shared between subsequent calls, you can
write the function like this instead::

   def f(a, L=None):
       if L is None:
           L = []
       L.append(a)
       return L


.. _tut-keywordargs:

Keyword Arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Functions can also be called using :term:`keyword arguments <keyword argument>`
of the form ``kwarg=value``.  For instance, the following function::

   def parrot(voltage, state='a stiff', action='voom', type='Norwegian Blue'):
       print("-- This parrot wouldn't", action, end=' ')
       print("if you put", voltage, "volts through it.")
       print("-- Lovely plumage, the", type)
       print("-- It's", state, "!")

accepts one required argument (``voltage``) and three optional arguments
(``state``, ``action``, and ``type``).  This function can be called in any
of the following ways::

   parrot(1000)                                          # 1 positional argument
   parrot(voltage=1000)                                  # 1 keyword argument
   parrot(voltage=1000000, action='VOOOOOM')             # 2 keyword arguments
   parrot(action='VOOOOOM', voltage=1000000)             # 2 keyword arguments
   parrot('a million', 'bereft of life', 'jump')         # 3 positional arguments
   parrot('a thousand', state='pushing up the daisies')  # 1 positional, 1 keyword

but all the following calls would be invalid::

   parrot()                     # required argument missing
   parrot(voltage=5.0, 'dead')  # non-keyword argument after a keyword argument
   parrot(110, voltage=220)     # duplicate value for the same argument
   parrot(actor='John Cleese')  # unknown keyword argument

In a function call, keyword arguments must follow positional arguments.
All the keyword arguments passed must match one of the arguments
accepted by the function (e.g. ``actor`` is not a valid argument for the
``parrot`` function), and their order is not important.  This also includes
non-optional arguments (e.g. ``parrot(voltage=1000)`` is valid too).
No argument may receive a value more than once.
Here's an example that fails due to this restriction::

   >>> def function(a):
   ...     pass
   ...
   >>> function(0, a=0)
   Traceback (most recent call last):
     File "<stdin>", line 1, in ?
   TypeError: function() got multiple values for keyword argument 'a'

When a final formal parameter of the form ``**name`` is present, it receives a
dictionary (see :ref:`typesmapping`) containing all keyword arguments except for
those corresponding to a formal parameter.  This may be combined with a formal
parameter of the form ``*name`` (described in the next subsection) which
receives a tuple containing the positional arguments beyond the formal parameter
list.  (``*name`` must occur before ``**name``.) For example, if we define a
function like this::

   def cheeseshop(kind, *arguments, **keywords):
       print("-- Do you have any", kind, "?")
       print("-- I'm sorry, we're all out of", kind)
       for arg in arguments:
           print(arg)
       print("-" * 40)
       keys = sorted(keywords.keys())
       for kw in keys:
           print(kw, ":", keywords[kw])

It could be called like this::

   cheeseshop("Limburger", "It's very runny, sir.",
              "It's really very, VERY runny, sir.",
              shopkeeper="Michael Palin",
              client="John Cleese",
              sketch="Cheese Shop Sketch")

and of course it would print::

   -- Do you have any Limburger ?
   -- I'm sorry, we're all out of Limburger
   It's very runny, sir.
   It's really very, VERY runny, sir.
   ----------------------------------------
   client : John Cleese
   shopkeeper : Michael Palin
   sketch : Cheese Shop Sketch

Note that the list of keyword argument names is created by sorting the result
of the keywords dictionary's ``keys()`` method before printing its contents;
if this is not done, the order in which the arguments are printed is undefined.

.. _tut-arbitraryargs:

Arbitrary Argument Lists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. index::
  statement: *

Finally, the least frequently used option is to specify that a function can be
called with an arbitrary number of arguments.  These arguments will be wrapped
up in a tuple (see :ref:`tut-tuples`).  Before the variable number of arguments,
zero or more normal arguments may occur. ::

   def write_multiple_items(file, separator, *args):
       file.write(separator.join(args))


Normally, these ``variadic`` arguments will be last in the list of formal
parameters, because they scoop up all remaining input arguments that are
passed to the function. Any formal parameters which occur after the ``*args``
parameter are 'keyword-only' arguments, meaning that they can only be used as
keywords rather than positional arguments. ::

   >>> def concat(*args, sep="/"):
   ...    return sep.join(args)
   ...
   >>> concat("earth", "mars", "venus")
   'earth/mars/venus'
   >>> concat("earth", "mars", "venus", sep=".")
   'earth.mars.venus'

.. _tut-unpacking-arguments:

Unpacking Argument Lists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The reverse situation occurs when the arguments are already in a list or tuple
but need to be unpacked for a function call requiring separate positional
arguments.  For instance, the built-in :func:`range` function expects separate
*start* and *stop* arguments.  If they are not available separately, write the
function call with the  ``*``\ -operator to unpack the arguments out of a list
or tuple::

   >>> list(range(3, 6))            # normal call with separate arguments
   [3, 4, 5]
   >>> args = [3, 6]
   >>> list(range(*args))            # call with arguments unpacked from a list
   [3, 4, 5]

.. index::
  statement: **

In the same fashion, dictionaries can deliver keyword arguments with the ``**``\
-operator::

   >>> def parrot(voltage, state='a stiff', action='voom'):
   ...     print("-- This parrot wouldn't", action, end=' ')
   ...     print("if you put", voltage, "volts through it.", end=' ')
   ...     print("E's", state, "!")
   ...
   >>> d = {"voltage": "four million", "state": "bleedin' demised", "action": "VOOM"}
   >>> parrot(**d)
   -- This parrot wouldn't VOOM if you put four million volts through it. E's bleedin' demised !


.. _tut-lambda:

Lambda Forms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By popular demand, a few features commonly found in functional programming
languages like Lisp have been added to Python.  With the :keyword:`lambda`
keyword, small anonymous functions can be created. Here's a function that
returns the sum of its two arguments: ``lambda a, b: a+b``.  Lambda forms can be
used wherever function objects are required.  They are syntactically restricted
to a single expression.  Semantically, they are just syntactic sugar for a
normal function definition.  Like nested function definitions, lambda forms can
reference variables from the containing scope::

   >>> def make_incrementor(n):
   ...     return lambda x: x + n
   ...
   >>> f = make_incrementor(42)
   >>> f(0)
   42
   >>> f(1)
   43


.. _tut-docstrings:

Documentation Strings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. index::
   single: docstrings
   single: documentation strings
   single: strings, documentation

Here are some conventions about the content and formatting of documentation
strings.

The first line should always be a short, concise summary of the object's
purpose.  For brevity, it should not explicitly state the object's name or type,
since these are available by other means (except if the name happens to be a
verb describing a function's operation).  This line should begin with a capital
letter and end with a period.

If there are more lines in the documentation string, the second line should be
blank, visually separating the summary from the rest of the description.  The
following lines should be one or more paragraphs describing the object's calling
conventions, its side effects, etc.

The Python parser does not strip indentation from multi-line string literals in
Python, so tools that process documentation have to strip indentation if
desired.  This is done using the following convention. The first non-blank line
*after* the first line of the string determines the amount of indentation for
the entire documentation string.  (We can't use the first line since it is
generally adjacent to the string's opening quotes so its indentation is not
apparent in the string literal.)  Whitespace "equivalent" to this indentation is
then stripped from the start of all lines of the string.  Lines that are
indented less should not occur, but if they occur all their leading whitespace
should be stripped.  Equivalence of whitespace should be tested after expansion
of tabs (to 8 spaces, normally).

Here is an example of a multi-line docstring::

   >>> def my_function():
   ...     """Do nothing, but document it.
   ...
   ...     No, really, it doesn't do anything.
   ...     """
   ...     pass
   ...
   >>> print(my_function.__doc__)
   Do nothing, but document it.

       No, really, it doesn't do anything.


.. _tut-codingstyle:
