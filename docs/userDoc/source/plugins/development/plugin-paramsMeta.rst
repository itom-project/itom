.. include:: ../../include/global.inc

.. sectionauthor:: Marc Gronle

.. _plugin-ParamsMeta:

Parameters - Meta Information
=============================

Class ParamMeta
---------------

The class **ParamMeta** is the base class for all meta information classes. Parameters of class **Param** may
contain pointers of that class, which then must be casted to the final implementation.

.. doxygenclass:: ito::ParamMeta
    :project: itom
    :members:

Class CharMeta, IntMeta and DoubleMeta
--------------------------------------

The classes **CharMeta**, **IntMeta** and **DoubleMeta** provide meta information for parameters of numeric types as well as their corresponding arrays.

.. doxygenclass:: ito::CharMeta
    :project: itom
    :members:

.. doxygenclass:: ito::IntMeta
    :project: itom
    :members:

.. doxygenclass:: ito::DoubleMeta
    :project: itom
    :members:

Class StringMeta
----------------

By this meta information you can give information about restrictions of strings to different strings. These strings can be interpreted as pure
strings, as wildcard-expressions or regular expressions. The corresponding checks must be defined manually. If a string-parameter has an enumeration defined,
where the strings are interpreted as strings, and if this parameter will automatically be parsed by any input mask in the GUI, the corresponding input text
box becomes a drop-down menu with the given enumeration elements.

.. doxygenclass:: ito::StringMeta
    :project: itom
    :members:

Class DObjMeta
--------------

This meta information class provides further information about allowed types and boundaries concerning the dimension of a data object.

.. doxygenclass:: ito::DObjMeta
    :project: itom
    :members:

Class HWMeta
------------

By that implementation of a meta information class you can provide information about references to other instantiated plugins.
Every plugin is defined by a bitmask of enumeration **ito::tPluginType** (defined in **addInActuator.h**). You can either add a minimum bitmask, that is required,
to the **HWMeta**-instance or you can define an exact name of a plugin, which must be met.

.. doxygenclass:: ito::HWMeta
    :project: itom
    :members:


