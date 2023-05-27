.. _plot-designer-plugins:

Programming plot designer plugins in C++
******************************************

General
===========================

Plots in itom are single libraries, programmed in C++ and based on the concept of **designer plugins** in Qt.
These designer plugins have the possibility to appear as widgets in the **Qt Designer**, if it is opened via itom
and can therefore be integrated in user-defined user interfaces.

itom designer plugins are located in the **designer** subfolder of itom. This folder will also be added as additional
search directory, if the Qt Designer application is started from itom (only in this case). Then the Qt designer also
scans this directory for compatible libraries, implemented the Qt designer plugin interface and - in the case of success -
makes the contained widgets available in the list of available widgets.


Dependencies and Versioning
============================

All designer plugins of itom (plots or other designer plugins) can or have to link against some libraries, contained
in the itom SDK. These libraries are (among others, see the folder **SDK/lib/<your compiler>**):

1. **itomCommonQtLib**: This library contains many basic classes for plugins, designer plugins...
2. **dataObject**: Contains the basic matrix class of itom, the :ref:`class dataObject <plugin-dataObject>`
3. **qpropertyeditor**: This class contains the property editor widget, that is used by all plots to show the properties of the plot
4. **itomWidgets**: This class is part of the itom core application and contains further smaller widgets, that can also be used in
    other plugins
5. **itomShapeLib**: This library contains the C++ library **ito::shape**, used as container for shapes like lines,
    rectangles, circles, squares, ellipses or polygons
6. **itomCommonLib**: This library contains further commonly used basic classes, which have no dependency to Qt (RetVal, Param...)
7. **itomCommonPlotLib**: This library contains further basic classes for plot plugins

Whenever itom is using a plot or other widget from an itom designer plugin, the binary compatibility between itom
and every plugin has to be assured (similar to other plugins of itom, see also :ref:`this document <plugins-loading>`).

This binary compatibility is defined by the following two main topics:

1. The plugin has to implement the correct interface, in accordance to its destination (plot plugin, further widget plugin, ...)
2. The plugin has to link against a binary compatible version of all libraries of the itom SDK, which are used by this plugin.

To assure this, the basic interfaces to plot plugins, designer plugins as well as general itom plugins are subject to
version numbers, that following the schematic of semantic versioning (semver.org). While the interface to general itom
plugins is versioned by the **addInInterface version number** (see the file **SDK/include/common/addInInterfaceVersion.h**),
the specific interface class to itom plot designer plugins, is versioned by the **itom designerplugin interface number**
(see the file **SDK/include/plot/designerPluginInterfaceVersion.h**).

In order to also cover the 2nd point of the binary compatibility list above, the **addInInterface version number** is
also incremented (following semantic versioning), if one of the libraries of the SDK are changed. Plot specific things
are covered by the **itom designerplugin interface number**.

.. note::

    itom will only load a designer plugin, if the two version numbers, readout from the library, are compatible to the
    required version numbers of the core application of itom. However, problems might occur if an incompatible designer
    widget library is located in the designer folder. Therefore, a message will be shown at startup of itom, that informs
    the user to manually delete affected files. The reason is, that itom cannot prevent the Qt Designer application, started
    from itom, to load these incompatible libraries, which might crash the Qt Designer application. The same crash might
    happen if the user generates a user-defined GUI from a *ui*-file, that contains at least one widget from an incompatible
    library.


Factory Class
==========================

The factory class of a designer plugin is usually derived from **QDesignerCustomWidgetInterface**. However if an itom
plot widget is implemented, its factory class has to be derived from the class **ito::AbstractItomDesignerPlugin**, which
itself is derived from **QDesignerCustomWidgetInterface**.

The header file of this factory class then looks like this:

.. code-block:: c++
    :linenos:

    #include "plot/AbstractItomDesignerPlugin.h"

    class YourPluginPlotFactory : public ito::AbstractItomDesignerPlugin
    {
        Q_OBJECT
        Q_PLUGIN_METADATA(IID "org.qt-project.Qt.QDesignerCustomWidgetInterface" FILE "pluginMetaData.json")

    public:
        YourPluginPlotFactory(QObject *parent = 0);

        bool isContainer() const;
        bool isInitialized() const;
        QIcon icon() const;
        QString domXml() const;
        QString group() const;
        QString includeFile() const;
        QString name() const;
        QString toolTip() const;
        QString whatsThis() const;
        QWidget *createWidget(QWidget *parent);
        QWidget *createWidgetWithMode(ito::AbstractFigure::WindowMode winMode, QWidget* parent);
        void initialize(QDesignerFormEditorInterface *core);

    private:
        bool initialized;
    };

Most of the methods follow the rules of default classes, implementing the **QDesignerCustomWidgetInterface** (see
https://doc.qt.io/qt-5/qdesignercustomwidgetinterface.html). However, there are some itom specific additions:

1. From itom 3.3.0, use the macro **Q_PLUGIN_METADATA** as stated in the snippet above. The indicated json file
   is an auto-generated json file, contained in the itom subfolder **SDK/include** and contains the two
   version numbers of the **itom addInInterface** and **itom designerplugin interface** (see chapter above). This
   meta information can then be read-out by itom at startup without the need to create an instances of contained
   classes. Please also add this macro to any other designer plugins, that make use of any libraries of the itom SDK.
2. **createWidgetWithMode**: This specific factory method is called only if a plot designer widget is called by
   a default itom figure (no user-defined GUI but an ordinary itom figure window).
