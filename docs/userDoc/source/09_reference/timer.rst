.. include:: ../include/global.inc

timer
*********

.. currentmodule:: itom

Currently, itom does not support the threading module from python. Therefore, no timed calls of python functions are possible using this module.
However, the :py:mod:`itom` provides the *timer* class to allow such calls:

    
timer-Class
============

.. autoclass:: itom.timer
    :member-order: groupwise
    :members: