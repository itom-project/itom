

.. include:: ../include/global.inc

Loops and Lists
===============
.. moduleauthor:: PSchau
.. sectionauthor:: PSchau



`for` Statements
------------------

The ``for`` statement in Python differs a bit from what you may be used to in C or Pascal. Rather than always iterating over an arithmetic progression of numbers, or giving the user the ability to define both the iteration step and halting condition, Python's ``for`` statement iterates over the items of any sequence (a list or a string), in the order that they appear in the sequence.  For example:

.. code-block:: python
    :linenos:

    # Measure some strings:
    words = ['cat', 'window', 'defenestrate']

    for w in words:
        print(w, len(w))

::

    cat 3
    window 6
    defenestrate 12

If you need to modify the sequence you are iterating over while inside the loop (for example to duplicate selected items), it is recommended that you first make a copy.  Iterating over a sequence does not implicitly make a copy. The slice notation makes this especially convenient:

.. code-block:: python
    :linenos:
    
    for w in words[:]:      # Loop over a slice copy of the entire list.
        if len(w) > 6:
            words.insert(0, w)
    words

::

    ['defenestrate', 'cat', 'window', 'defenestrate']


The range Function
--------------------

If you do need to iterate over a sequence of numbers, the built-in function :py:func:`range` comes in handy. It generates arithmetic progressions:

.. code-block:: python
    :linenos:

    for i in range(5):
        print(i)


The given end point is never part of the generated sequence; ``range(10)`` generates 10 values, the legal indices for items of a sequence of length 10. It is possible to let the range start at another number, or to specify a different increment (even negative; sometimes this is called the 'step'):

.. code-block:: python
    :linenos:
    
    range(5, 10)                # 5 through 9
    range(0, 10, 3)             # 0, 3, 6, 9
    range(-10, -100, -30)       # -10, -40, -70

To iterate over the indices of a sequence, you can combine :py:func:`range` and :py:func:`len` as follows:

.. code-block:: python
    :linenos:
    
    a = ['Mary', 'had', 'a', 'little', 'lamb']
    for i in range(len(a)):
        print(i, a[i])

::

    0 Mary
    1 had
    2 a
    3 little
    4 lamb

In most such cases, however, it is convenient to use the :py:func:`enumerate` function.


`break` and `continue` Statements, and `else` Clauses on Loops
------------------------------------------------------------------

The `break` statement, like in C, breaks out of the smallest enclosing `for` or `while` loop.

Loop statements may have an ``else`` clause; it is executed when the loop terminates through exhaustion of the list (with `for`) or when the condition becomes false (with `while`), but not when the loop is terminated by a `break` statement.  This is exemplified by the following loop, which searches for prime numbers:

.. code-block:: python
    :linenos:

    for n in range(2, 10):
        for x in range(2, n):
            if n % x == 0:
                print(n, 'equals', x, '*', n//x)
                break
            else:
                # loop fell through without finding a factor
                print(n, 'is a prime number')

::

    2 is a prime number
    3 is a prime number
    4 equals 2 * 2
    5 is a prime number
    6 equals 2 * 3
    7 is a prime number
    8 equals 2 * 4
    9 equals 3 * 3

When used with a loop, the ``else`` clause has more in common with the ``else`` clause of a `try` statement than it does that of `if` statements: a `try` statement's ``else`` clause runs when no exception occurs, and a loop's ``else`` clause runs when no ``break`` occurs. For more on the `try` statement and exceptions.

The `continue` statement continues with the next iteration of the loop:

.. code-block:: python
    :linenos:
    
    for num in range(2, 10):
        if num % 2 == 0:
            print("Found an even number", num)
            continue
        print("Found a number", num)

::

    Found an even number 2
    Found a number 3
    Found an even number 4
    Found a number 5
    Found an even number 6
    Found a number 7
    Found an even number 8
    Found a number 9
