

.. include:: ../include/global.inc

If and else statements
=======================
.. moduleauthor:: PSchau
.. sectionauthor:: PSchau




`if` Statements
-----------------
Perhaps the most well-known statement type is the `if` statement. For example:

.. code-block:: python
    :linenos:
    
    x = int(input("Please enter an integer: "))
    
    if x < 0:
        print('Negative')
    elif x == 0:
        print('Zero')
    else:
        print('Positive')

There can be zero or more `elif` parts, and the `else` part is optional.  The keyword '`elif`' is short for 'else if', and is useful to avoid excessive indentation.  An `if` ... `elif` ... `elif` ... sequence is a substitute for the ``switch`` or ``case`` statements found in other languages.