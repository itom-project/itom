

.. include:: ../include/global.inc

Numbers and math
==================
.. moduleauthor:: PSchau
.. sectionauthor:: PSchau




Python arithmetic operators:
------------------------------

Here is an overview about the basic math operators and how they are used in |Pythonv3|. You can type simple math equations (e.g. the examples from the table below) directly into the |itom| terminal to see how some of the operators work. Alternatively, you can create yourself a new |Python| script file and type in some math there.

+-----------+----------------------------+-----------------------------------------------------------------+
| Operator  | Description                | Example                                                         |
+===========+============================+=================================================================+
| ``+``     | Addition                   | 10 + 20 will give 30                                            |
+-----------+----------------------------+-----------------------------------------------------------------+
| ``-``     | Subtraction                | 10 - 20 will give -10                                           |
+-----------+----------------------------+-----------------------------------------------------------------+
| ``*``     | Multiplication             | 10 * 20 will give 200                                           |
+-----------+----------------------------+-----------------------------------------------------------------+
| ``/``     | Division                   | 20 / 10 will give 2                                             |
+-----------+----------------------------+-----------------------------------------------------------------+
| ``%``     | Modulus                    | 20 % 10 will give 0                                             |
+-----------+----------------------------+-----------------------------------------------------------------+
| ``**``    | Exponent                   | ``10**2`` will give ``10`` to the power ``2``                   |
+-----------+----------------------------+-----------------------------------------------------------------+
| ``//``    | Floor division             | ``9//2`` is equal to ``4`` and ``9.0//2.0`` is equal to ``4.0`` |
+-----------+----------------------------+-----------------------------------------------------------------+


Python comparison/boolean operators:
---------------------------------------

In addition, comparison operators are shown in the following table. As displayed in the example column, they mainly serve for the comparison of two numbers and return a boolean value - *true* or *false*.

+-----------+----------------------------+----------------------------------------------------------------+
| Operator  | Description                | Example                                                        |
+===========+============================+================================================================+
| ``==``    | Equal                      | (10 == 20) is not true                                         |
+-----------+----------------------------+----------------------------------------------------------------+
| ``!=``    | Not equal                  | (10 != 20) is true                                             |
+-----------+----------------------------+----------------------------------------------------------------+
| ``<>``    | Not equal                  | (10 <> 20) is true                                             |
+-----------+----------------------------+----------------------------------------------------------------+
| ``<``     | Less-than                  | (10 < 20) is true                                              |
+-----------+----------------------------+----------------------------------------------------------------+
| ``>``     | Greater-than               | (10 > 20) is not true                                          |
+-----------+----------------------------+----------------------------------------------------------------+
| ``<=``    | Less-than-equal            | (10 <= 20) is true                                             |
+-----------+----------------------------+----------------------------------------------------------------+
| ``>=``    | Greater-than-equal         | (10 >= 20) is not true                                         |
+-----------+----------------------------+----------------------------------------------------------------+


More examples
----------------

Here are a few more examples to practice for yourself how the different operators work. Type in the following source code in another script file and run the code to see what happens.

.. code-block:: python
    :linenos:

    # Python calculates with the correct order of operations
    print(25 + 30 / 6)
    print(100 - 25 * 3 % 4)
    print(3 + 2 + 1 - 5 + 4 % 2 - 1 / 4 + 6)

    # Difference between regular division (/) and floor division (//)
    print(8/5)
    print(8//5)

::

    30.0
    97
    6.75
    1.6
    1

You can also combine your math/arithmetic operations with the :py:func:`print` command

.. code-block:: python
    :linenos:

    print("Is it true that 3 + 2 < 5 - 7?")     # outputs all characters in between the quotation marks
    print(3 + 2 < 5 - 7)                        # prints the actual result

    print("Is 5 greater than 7?", 5 > -2)       # combines the previous example in one line

    print("What is 3 + 2?", 3 + 2 )

::

    Is it true that 3 + 2 < 5 - 7?
    False
    Is 5 greater than 7? True
    What is 3 + 2? 5

As you might have noticed, lines starting with the hash character ``#`` are not interpreted by |Python|. In this fashion, the programmer can render his file easier understandable for another programmer or for himself when coming back to a complex program after a while. Comments can also start behind a line of code. They extend until the end of the physical line.

Also, you can combine the :py:func:`print` command with mathematic/arithmetic operations, which is explained in more detail in chapter :ref:`Strings and text <pytut_strings_text>`.

.. seealso:: You can find even more math functions in the `official documentation <http://docs.python.org/py3k/library/math.html>`_.
