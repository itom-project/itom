.. include:: /include/global.inc

.. _plugins:

Plugins
###################

This chapter contains information about the software and hardware plugin mechanism of |itom|. The software plugins may contain algorithms, written in C++, which can
be called from any python script or another plugin. Furthermore, software plugins can also contain arbitrary user interfaces. This allows implementing complex dialogs
and windows using all possibilities given by the Qt-framework. Hardware plugins allow to implement actuators, cameras, AD-converters and other devices. Then these devices
are also accessible by any python script.

See the following sections in order to get more information about these plugins:

.. toctree::
    :maxdepth: 1

    basic_concept.rst
    getstart-hardware.rst
    getstart-grabber.rst
    getstart-adda.rst
    getstart-actuator.rst
    getstart-filter.rst
    
If you want to program your own plugin, see the following sections of the documentation:

.. toctree::
    :maxdepth: 2
   
    development/plugin-development.rst

 
