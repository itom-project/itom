.. include:: ../include/global.inc

.. _algoAndWidgets:

Algorithms and Widgets
*********************************

.. currentmodule:: itom

itom algorithm plugins can contain two different types of code:

1. Most algorithm plugins consist of one or multiple algorithms (also denoted as filters)
   that can both be called from Python scripts as well as from any other itom plugin.

   To call an algorithm with the exemplary name ``myAlgo`` from Python, use one of
   the two following possibilities:

   * call ``itom.filter("myAlgo", *args, **kwds)`` for a generic call of any algorithm.
   * use the wrapper method ``myAlgo`` from the submodule ``algorithms`` of ``itom``:

     .. code-block:: python

        import itom
        # args and kwds are the mandatory (or optional) parameters of the
        # specific algorithm 'myAlgo'. The returned value is either None,
        # a single value or a tuple of values.
        result = itom.algorithms.myAlgo(*args, **kwds)

     This 2nd possibility is available from itom 5.0.0 on and allows using auto completion
     and calltips with a context sensitive help for the specific algorithm. The set of
     available methods in the ``itom.algorithms`` submodule depends on the loaded algorithm
     plugins of your itom application.

2. Algorithm plugins can also contain custom user interfaces. The create and show such
   an user interface, call the method :py:meth:`~itom.ui.createNewPluginWidget` or
   :py:meth:`~itom.ui.createNewPluginWidget2`. To get an online help, use :py:meth:`~itom.widgetHelp`.

For more information, see also :ref:`getStartFilter` or in the demo file **demoGaussianSpotCentroidDetection.py**.

.. automodule:: itom
    :members: filter, filterHelp, widgetHelp
