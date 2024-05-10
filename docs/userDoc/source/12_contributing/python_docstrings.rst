.. _python_docstrings:

Python Docstrings
************************

General
============

For Python docstrings, we have to differentiate between docstrings in real Python script files (.py) or
docstrings in **C-Extensions**, like it is the case for the entire **itom** module.

Docstrings for :py:mod:`itom` and its methods and classes
==========================================================

Before we start with the style guide for the docstrings, a short introduction about the workflow
and usage of these docstrings is given.

The docstrings are mainly used for three different use cases:

    1. Text, that appears if one wants to get help using the :py:meth:`help` command of python
       or if the ``__doc__`` attribute is manually requested by the user.
    2. The script reference of the :py:mod:`itom` is automatically parsed by the ``autodoc``
       feature of ``Sphinx``. Autodoc directly parses the docstrings and must be able to
       parse the entire signature, including type hints, the docstring as well as the description
       of parameters and return values from the docstring. For this, itom docstrings should be
       based on the Sphinx extension **numpydoc** (see below). Sphinx is not able to parse any
       further stubs files (.pyi).
    3. For calltips, auto completion etc. the optional package **Jedi** is used. It also tries
       to parse the signature, arguments, their types, the return types etc. from the docstring.
       However, since the source of the :py:mod:`itom` module is only available as compiled code,
       Jedi has trouble to get all these information. Therefore, itom helps **Jedi** by providing
       a pseudo-code stubs file, located in the file ``itom-packages/itom-stubs/__init__.pyi``.
       This file is automatically parsed from the original docstrings of all methods and classes
       of the :py:mod:`itom` module.

In order to support these three use cases (mainly no 2 and 3), this guide shows how to
write docstrings in the **C/C++** source code for methods and classes, that belong
to the :py:mod:`itom` module.

General rules
--------------

1. Always use the macro ``PyDoc_STRVAR`` to generate the docstring
2. Do not indent multiline docstrings in **C/C++**, since this indentation is kept in the docstring.
   Therefore, always start these multiline docstrings in the first column of the source file!
3. Avoid long lines in the docstrings. Wrap them with an explicit newline character (\n) after a
   maximum number of 88 characters.
4. Use four spaces as indentation level
5. Most parts of the docstrings should follow the rules of
   `Numpydoc <https://pypi.org/project/numpydoc/>`_. For more information see
   `https://numpydoc.readthedocs.io/en/latest/format.html <https://numpydoc.readthedocs.io/en/latest/format.html>`_
   (The Numpydoc package must only be installed if you want to rebuild the Sphinx user documentation).

Method docstrings
-----------------

The docstring of an unbounded method looks like this

.. code-block:: C

    PyDoc_STRVAR(varname_doc,
    "myMethod(arg1, arg2 = -2, arg3 = None) -> List[int] \n\
    \n\
    short description of this method. \n\
    \n\
    longer multiline description of this method. \n\
    Emphasize some word like ``this``. Or make a \n\
    reference to another method like :meth:`openScript`. \n\
    \n\
    Parameters \n\
    ---------- \n\
    arg1 : int \n\
        description of arg1. \n\
    arg2 : int, optional \n\
        description of the optional argument 2. \n\
    arg3 : str or None, optional \n\
        longer description of ``arg3``. \n\
        Maybe go to a 2nd line. \n\
    \n\
    Returns \n\
    ------- \n\
    list of int \n\
        description of the return value, \n\
        that can also be multiline. \n\
    \n\
    Raises \n\
    ------ \n\
    RuntimeError \n\
        if a certain situation occurs. \n\
    \n\
    See Also \n\
    -------- \n\
    openScript, newScript, ...");

In the first line, the signature of the method should be given. Write
this signature, like you would do it in an ordinary python script. It is
possible to add type hints there for the arguments, however this can also
be done within the ``Parameters`` section below, where the parameters are
described further.

If the method has no return value, omit the ``-> ret-type`` at the end of
the signature line. Else, it is allowed to write the return type here, following
the rules of the :mod:`typing` module of Python. The type hint is recommended, but
optional. It can also be obtained from the ``Returns`` section. We recommend both.

After the signature line, write a short description of the method after a separate and empty
new line. Then, insert another new line and continue with a multiline long
description of the method. This long description can also consist of further
sections, following the rules of Numpydoc. If the method has at least one
argument, it is recommend to describe it in a ``Parameters`` section. If the
method has a return value, use the ``Returns`` section. If you want to add a
reference to other methods, use the ``See Also`` section.

.. note::

    Please consider, that the underline of the sections must be at least as long as
    the name of the section. Hence, a ``Parameters`` section must be followed by
    the underline line with at least 10 ``-`` characters.

Please consider the following rules for type hints:

* If you write any type hints in the signature line, always use the type hints as
  given by the :mod:`typing` module of Python. Examples are: ``Optional[int]``,
  ``str``, ``Union[Tuple[int, str]]`` among others.
* If you write any type hints in the docstrings section, follow the rules of
  **Numpydoc**. Examples are then: ``int or None``, ``str``, ``tuple of int or None``.
  Values like ``tuple of int`` will be transformed to a type hint ``Tuple[int]``,
  hence, the first value before the ``of`` literal will be written with a capital
  letter, the 2nd argument will be put into square brackets. Several type
  possibilities, separated by the ``or`` literal in Numpydoc will be transformed
  to ``Union[a, b]`` if the Numpydoc typehint was ``a or b``. If the Numpydoc
  typestring ends with ``abc, optional``, its typing equivalent is ``Optional[abc]``.
  If you want to use the type ``Any`` from typing, you also have to write ``Any``
  in the Numpydoc section.

Overloaded methods
------------------

It is also possible to support overloaded methods, that accept different
sets of parameters. If this is the case, write all possible signatures in
the first lines and write a backslash as last character (no spaces afterwards)
of the first signature lines (all signature lines beside the last one).
Please be aware, that this backslash must be written in the C-code by two
backslashes.

.. note::

    If overloaded methods are added, every signature needs to have a return type.
    This is maybe a bug in Sphinx, however with Sphinx 3.3 this was the case.

Use the ``Parameters`` section to explain all arguments, even if they are
only used in one of the signatures.

Here is an example for this:

.. code-block:: C

    PyDoc_STRVAR(varname_doc,
    "myMethod() -> int \\\n\
    myMethod(number) -> int \\\n\
    myMethod(text) -> int \n\
    \n\
    short description of this method. \n\
    \n\
    longer multiline description of this method. \n\
    \n\
    Parameters \n\
    ---------- \n\
    number : int \n\
        docstring for ``number``. \n\
    text : int \n\
        docstring for ``text``. \n\
    \n\
    Returns \n\
    ------- \n\
    name : int \n\
        a return value can also have a name (optional).");

Classes and constructor
-----------------------

The C-Extension does not provide a simple possibility to add
a docstring to the ``__init__`` method only. Therefore, the
general class docstring, passed to the ``tp_doc`` member of
the ``PyTypeObject`` struct should contain both a description of
the entire class as well as the signature of its constructor and
the parameters (exceptions, ...).

The following example shows the docstring for the class :class:`~itom.region`,
whose signature has three different overloads (see again the backslash at the
end of the first signatures). The method name of the signature is the class
name, not the literal ``__init__``:

.. code-block: C

    PyDoc_STRVAR(pyRegion_doc,"region() -> region \\\n\
    region(otherRegion) -> region \\\n\
    region(x, y, w, h, type = region.RECTANGLE) -> region \n\
    \n\
    Creates a rectangular or elliptical region. \n\
    \n\
    This class is a wrapper for the class ``QRegion`` of `Qt`. It provides possibilities for \n\
    creating pixel-based regions. Furtherone you can calculate new regions based on the \n\
    intersection, union or subtraction of other regions. Based on the region it is \n\
    possible to get a uint8 masked dataObject, where every point within the entire \n\
    region has the value 255 and all other values 0 \n\
    \n\
    If the constructor is called without argument, an empty region is created. \n\
    \n\
    Parameters \n\
    ----------- \n\
    otherRegion : region \n\
        Pass this object of :class:`region` to create a copied object of it. \n\
    x : int\n\
        x-coordinate of the reference corner of the region \n\
    y : int\n\
        y-coordinate of the reference corner of the region \n\
    w : int\n\
        width of the region \n\
    h : int\n\
        height of the region \n\
    type : int, optional \n\
        ``region.RECTANGLE`` creates a rectangular region (default). \n\
        ``region.ELLIPSE`` creates an elliptical region, which is placed inside of the \n\
        given boundaries.");

Properties
-----------

The docstring for the properties (@property decorator) is usually
related to the getter method. If a setter is available, try to
consider this in the longer description part of the docstring.

The return type of the getter property must be written in the
Numpydoc style, e.g. ``list of tuple of int`` or ``int or None``.

The following examples show, that no signature is contained in
the docstrings. The docstring start with the return type hint,
followed by a colon, a space character and a short
description of the property. An optional longer, multiline description
can be added after a newline.

Examples are:

.. code-block:: C

    PyDoc_STRVAR(property1_doc,
    "list of list of int: Short description comes here. \n\
    \n\
    Optional longer description (can be multiline)");


.. code-block:: C

    PyDoc_STRVAR(property1_doc,
    "str or None: Short description comes here.");



References
==================

This style guide is mainly taken from

* Numpydoc (https://numpydoc.readthedocs.io/en/latest/format.html)
* Napoleon extension of Sphinx (https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
* Numpydoc example of Napoleon extension (https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html#example-numpy)
