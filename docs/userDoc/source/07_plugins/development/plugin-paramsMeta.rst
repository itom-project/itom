.. include:: ../../include/global.inc

.. sectionauthor:: Marc Gronle

.. _plugin-ParamsMeta:

Parameters - Meta Information
=============================

Every parameter of type ito::Param can contain meta information that describe some boundary values, value ranges, 
allowed values... of the parameter. Once a parameter has its valid meta information, itom is able to check given 
input values with respect to the meta information as well as adapt any auto-created input masks to simplify the 
input with respect to the given constraints.

Most possible types of class ito::Param have their respective meta information structure.

.. _plugin-paramMeta-scheme:
.. figure:: images/paramMeta.png
    :scale: 100%
    :align: center

The base class of all kind of meta information classes is the class :ref:`ito::ParamMeta <classParamMeta>`. Any 
object of class :ref:`ito::Param <classParam>` can contain a pointer to an instance of this base class. 
If you know the type of parameter (e.g. char, string or int), you can safely cast this :ref:`ito::ParamMeta 
<classParamMeta>` base instance to the right meta information class that fits to the type. The type
can also be requested at runtime by the RTTI type information in ParamMeta.

.. _classParamMeta:

Class ParamMeta
-----------------

The class **ParamMeta** is the base class for all meta information classes. Parameters of class :ref:`ito::Param 
<classParam>` may contain pointers of that class, which then must be cast to the final implementation.

.. doxygenclass:: ito::ParamMeta
    :project: itom
    :members:

Class CharMeta, IntMeta and DoubleMeta
------------------------------------------

The classes **CharMeta**, **IntMeta** and **DoubleMeta** provide meta information for parameters of single 
numeric types.

.. doxygenclass:: ito::CharMeta
    :project: itom
    :members:

.. doxygenclass:: ito::IntMeta
    :project: itom
    :members:

.. doxygenclass:: ito::DoubleMeta
    :project: itom
    :members:

Currently, there is no meta object for **Complex** parameter types.
    
Class CharArrayMeta, IntArrayMeta and DoubleArrayMeta
--------------------------------------------------------

The classes **CharArrayMeta**, **IntArrayMeta** and **DoubleArrayMeta** provide meta information for array-based 
parameters of numeric types. These classes are derived from **CharArray**, **IntArray** or **DoubleArray**, such 
that the minimum and maximum value as well as the step size for each single value is given by the features of 
their base class. Additionally, it is possible to set a min, max and stepSize constraint concerning the number of 
elements of the arrays.

.. doxygenclass:: ito::CharArrayMeta
    :project: itom
    :members:

.. doxygenclass:: ito::IntArrayMeta
    :project: itom
    :members:

.. doxygenclass:: ito::DoubleArrayMeta
    :project: itom
    :members:

Class StringMeta
------------------

This meta information object can be assigned to String parameters. By this, it is possible to choose between
a certain type of string contraints:

1. Wildcard expressions: Pass one wildcard expression to the meta information. The string parameter
   is then checked against this wildcard expression, e.g. **image_??.jpg**.
2. Regular expression: The same than Wildcard, however the string is a regular expression.
3. Fixed: You can pass one or multiple strings. The given string parameter will then be checked to
   exactly match one of the strings in this **StringMeta**.

If such a string parameter is input by a dialog in itom, the appearance of the input field depends one
the types. Usually it is a default text box, which might have some input restrictions if the type is
Wildcard or Regular Expression. In the case of Fixed strings, a combobox appears with all possible strings.

.. doxygenclass:: ito::StringMeta
    :project: itom
    :members:

Class StringListMeta
----------------------

A StringListMeta can decorate an **ito::Param** object of type **StringList**. It is derived from **StringMeta**
and can further hold a minimum and maximum number of allowed values in the string list, as well as a certain
step size of the number of elements.

.. doxygenclass:: ito::StringListMeta
    :project: itom
    :members:

Class IntervalMeta, Class RangeMeta
-------------------------------------

An interval meta can decorate an **Integer-Array** parameter whith two values. It is derived from **IntMeta**,
that describes the minimum and maximum value, as well as the step size, of both limits of the interval.

If the first value in the array is denoted as **val0** and the 2nd one **val1**, they are checked
to meet the following condition:

**min <= val0 <= val1 <= max**.

By IntervalMeta or RangeMeta you can further constraint the size of the interval or range. The size
is defined by two different possible methods:

* **IntervalMeta**: *size = val1 - val0*
* **RangeMeta**: *size = 1 + val1 - val0*

For a **RangeMeta** object, both the first and 2nd value are part of the range, for intervals, the
last value is not contained. RangeMeta is usually used for region of interests of cameras, usually
using the class **RectMeta**, that consists of two **RangeMeta** for the horizontal and vertical direction.

.. doxygenclass:: ito::IntervalMeta
    :project: itom
    :members:

.. doxygenclass:: ito::RangeMeta
    :project: itom
    :members:

Class RectMeta
---------------

The RectMeta can decorate an **Integer-Array** parameter with exactly four integer values, whose
definition is **x0, y0, width, height**. To constraint both the limits as well as the size of
both directions (horizontal, vertical), RectMeta contains two **RangeMeta** members.

.. doxygenclass:: ito::RectMeta
    :project: itom
    :members:

Class DoubleIntervalMeta
-------------------------

This meta object decorates a **Double-Array** parameter with two values and is similar to **IntervalMeta**.
If the **sizeStepSize** is equal to 0.0, the size of the interval can have any value between
**sizeMin** and **sizeMax**.

.. doxygenclass:: ito::DoubleIntervalMeta
    :project: itom
    :members:

Class DObjMeta
----------------

This meta information class provides further information about allowed types and boundaries concerning the 
dimension of a data object.

.. doxygenclass:: ito::DObjMeta
    :project: itom
    :members:

Class HWMeta
--------------

By that implementation of a meta information class you can provide information about references to other 
instantiated plugins. Every plugin is defined by a bitmask of enumeration **ito::tPluginType** (defined in 
**addInActuator.h**). You can either add a minimum bitmask, that is required, to the **HWMeta**-instance or you 
can define an exact name of a plugin, which must be met.

.. doxygenclass:: ito::HWMeta
    :project: itom
    :members:


