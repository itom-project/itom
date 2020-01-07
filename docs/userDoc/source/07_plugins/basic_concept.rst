.. include:: /include/global.inc

Basic concept
**************************************

Obviously the power of  |itom| comes from the embedded scripting language |Python| that is enhanced by a graphical 
user interface and many other features programmed in C++. |itom| should be able to either communicate with various 
hardware devices, like cameras (grabbers), AD-DA converters, communication protocols or actuators. On the other hand, 
the acquired data has to be analyzed, processed and visualized.

In order to provide a fast access to these devices, it is necessary to link to them by specific SDKs. All this is done 
via plugins in |itom|, hence every hardware device is wrapped by an |itom| plugin that is loaded at startup. 
Nevertheless, it is not desired to have plugins with arbitrary interfaces, such that one has to find the right command 
set and communcation strategy that differs from camera to camera (this is maybe a little bit the case in Matlab or 
LabView, where every manufacturer can implement its plugin with its own function-set and communication strategy). 
|itom| provides a unified interface for the three device categories: **dataIO** (for all grabbers and other input/output 
devices), **actuator** (for actuators and stages) as well as an interface for algorithm plugins that allow a fast 
implementation using C/C++, CUDA, OpenMP, OpenCL or other 3rd Party libraries. This interface is either callable by
other C++ programmed plugins or by the |Python| scripting language.

Beside the script-based access to plugins, they can also offer configuration dialogs or a toolbox to easily control 
the device by the GUI of |itom|. The unified interface also allows implementing further generic structures, that can 
be called or operated with any camera or actuator without a need of rewritting parts of the code.

See this section for more information about how to use hardware and software plugins.