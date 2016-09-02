

.. include:: ../include/global.inc

.. _pytut_strings_text:

Strings and text 
=================
.. moduleauthor:: PSchau
.. sectionauthor:: PSchau


Strings
---------

Besides numbers, |Python| can also manipulate strings, which can be expressed in several ways. A string is usually a bit of text you want to display to someone, or "export" out of the program you are writing. They can be enclosed in single quotes or double quotes:

.. code-block:: python
    :linenos:
    
    print('It works.')
    print('It doesn\'t matter.')
    print("It doesn't matter.")
    print('"Yes", he said.')
    print("\"Yes,\" he said.")
    print('"It\'s not", she said.')

::

    It works.
    It doesn't matter.
    It doesn't matter.
    "Yes", he said.
    "Yes," he said.
    "It's not", she said.


String formatting
-------------------

Additionally, Strings may contain format characters such as ``%d`` from the previous section to output or convert to integer decimals. Here are some more:

+---------------------+-----------------------------------------------------------------------------------------+
| Conversion          | Meaning                                                                                 |
+=====================+=========================================================================================+
| ``%d`` or ``%i``    | Signed integer decimal                                                                  |
+---------------------+-----------------------------------------------------------------------------------------+
| ``%x`` or ``%X``    | Signed hexadecimal (lowercase/uppercase)                                                |
+---------------------+-----------------------------------------------------------------------------------------+
| ``%e`` or ``%E``    | Floating point exponential (lowercase/uppercase)                                        |
+---------------------+-----------------------------------------------------------------------------------------+
| ``%f`` or ``%F``    | Floating point decimal format                                                           |
+---------------------+-----------------------------------------------------------------------------------------+
| ``%c``              | Single character (accepts integer or single character string)                           |
+---------------------+-----------------------------------------------------------------------------------------+
| ``%r``              | String (converts any Python object using :py:func:`repr`)                               |
+---------------------+-----------------------------------------------------------------------------------------+
| ``%s``              | String (converts any Python object using :py:func:`str`)                                |
+---------------------+-----------------------------------------------------------------------------------------+

Here are some examples to try for yourself:

.. code-block:: python
    :linenos:
    
    x = "There are %d types of people." % 10
    binary = "binary"
    do_not = "don't"
    y = "Those who know %s and those who %s." % (binary, do_not)

    print(x)
    print(y)
    
    print("I said: %r." % x)
    print("I also said: '%s'." % y)

::

    There are 10 types of people.
    Those who know binary and those who don't.

.. note:: The formatting operations described here exhibit a variety of quirks that lead to a number of common errors (such as failing to display tuples and dictionaries correctly). Using the newer :py:meth:`str.format` interface helps avoiding these errors. Check the official `documentation <http://docs.python.org/py3k/library/stdtypes.html#index-30>`_ for more on the topic.


More on strings
------------------

String literals can span multiple lines in several ways. Continuation lines can be used, with a backslash as the last character on the line indicating that the next line is a logical continuation of the line:

.. code-block:: python
    :linenos:
    
    hello = "This is a rather long string containing\n\
    several lines of text just as you would do in C.\n\
        Note that whitespace at the beginning of the line is \
    significant."
    
    print(hello)

::

    This is a rather long string containing
    several lines of text just as you would do in C.
    Note that whitespace at the beginning of the line is significant.

Or, strings can be surrounded in a pair of matching triple-quotes: ``"""`` or ``'''``. End of lines do not need to be escaped when using triple-quotes, but they will be included in the string. So the following uses one escape to avoid an unwanted initial blank line.

.. code-block:: python
    :linenos:
    
    print("""\
    Usage: thingy [OPTIONS]
        -h                        Display this usage message
        -H hostname               Hostname to connect to
    """)

::

    Usage: thingy [OPTIONS]
         -h                        Display this usage message
         -H hostname               Hostname to connect to

If we make the string literal a "raw" string, \n sequences are not converted to newlines, but the backslash at the end of the line, and the newline character in the source, are both included in the string as data. Thus, the example::

    hello = r"This is a rather long string containing\n\
    several lines of text much as you would do in C."
    
    print(hello)

::
    
    "This is a rather long string containing\n\several lines of text much as you would do in C."
    
    
Byte array....
---------------

:)