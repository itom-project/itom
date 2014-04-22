.. include:: ../../include/global.inc

.. sectionauthor:: Alexander Bielke

.. _plugin-class-dataio:

Plugin class - DataIO
=========================

Base idea behind any DataIO
-----------------------------

* The DataIO operates with plugged systems require input and output, in generally cameras and frame grabbers but also serial ports or AD-DA-converter.
* One opened DataIO handles only one connected device. More cameras or ADDA-converter of the same type should be managed with a corresponding number of DataIO-ports.
* Depending on the plugin, more than one DataIO can be separately controlled by the plug-number or the vendor/camera-ID.
* Like any other plugin, every DataIO has a set of parameters, which can be set/get using python by the commands **setParam** and **getParam**.
* Every DataIO is executed in its own thread.
* Every DataIO can have one configuration dialog and one docking widget toolbox.
* All parameters are stored in **m_params**. They can be read or changed by methods **getParam** and **setParam**. Some parameters are read only!
* If parameters of DataIO changes, they also must be updated in the **m_params**-map.
* To connection between the dataIO-device (e.g. the camera) is done within the c++-method **init**. The initialization parameter should be given to **m_params**. In the end it must be disconnected by method **close**.
* The data acquisition is performed according to the grabber subtype. These subtypes are 'typeGrabber', 'typeADDA' and 'typeRawIO'.

Grabber plugin
------------------------------

This is a subtype of DataIO for camera / framegrabber communication. Plugins of this type are inherited from **ito::AddInGrabber**. The data acquisition is managed as follows:

* The methods **startDevice** and **stopDevice** opens and closes the capture logic of the devices to reduce CPU-load. For serial ports these functions are unnecessary. 
* The method **acquire** starts the DataIO grabbing a frame with the current parameters. The function returns after sending the trigger. The function should be callable several times without calling get-/copyVal().
* The methods **getVal** and **copyVal** are the external interfaces for data grabbing. They call **retrieveData**. The function should not be callable without a previous call of **acquire** and than only once.
* In **retrieveData** the data transfer is done and frame has to copied. The function blocks until the triggered data is copied. In case retrieveData is called by getVal the frame has to be copied to **m_data**, an internal **dataObject**.
* The function **getVal** overwrites the IO-**dataObject** by a shallow copy of the internal **dataObject**. Empty objects are allowed. Warning read shallow copy of dataObject before usage.  
* The function **copyVal** deeply copies data to the externally given **dataObject**. The **dataObject** must have the right size and type. **dataObject** with ROI must not be overwritten. The ROI should be filled. Empty objects are allowed. In case of empty **dataObject** a new object with right size and type must be allocated.
* The internal **dataObject** is checked after parameter changes by **checkData** (sizex, sizey and bpp) and, if necessary, reallocated.

A typical sequence in python is 

.. code-block:: python
    :linenos:
    
    device.startDevice()
    device.acquire()
    device.getVal(dObj)
    device.acquire()
    device.getVal(dObj)
    device.stopDevice()

    
A sample header file of the DataIO's plugin class is illustrated in the following code snippet:

.. code-block:: c++
    :linenos:

    #include "common/addInGrabber.h"
    #include <qsharedpointer.h>

    class MyCamera : public ito::AddInGrabber
    {
        Q_OBJECT

        protected:
            ~MyDataIO(); /*! < Destructor*/
            MyDataIO();    /*! < Constructor*/

            ito::RetVal retrieveData(ito::DataObject *externalDataObject = NULL); /*!< Wait for acquired picture */

        public:
            friend class MyDataIOInterface;
            const ito::RetVal showConfDialog(void);    //! Open the config nonmodal dialog to set camera parameters 
            int hasConfDialog(void) { return 1; }; //!< indicates that this plugin has got a configuration dialog

        private:

        public slots:
            ito::RetVal getParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore *waitCond);
            ito::RetVal setParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore *waitCond);

            ito::RetVal init(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, ItomSharedSemaphore *waitCond = NULL);
            ito::RetVal close(ItomSharedSemaphore *waitCond);

            ito::RetVal startDevice(ItomSharedSemaphore *waitCond);
            ito::RetVal stopDevice(ItomSharedSemaphore *waitCond);

            ito::RetVal acquire(const int trigger, ItomSharedSemaphore *waitCond = NULL);

            ito::RetVal getVal(void *vpdObj, ItomSharedSemaphore *waitCond);

            ito::RetVal copyVal(void *vpdObj, ItomSharedSemaphore *waitCond);

        private slots:
            void dockWidgetVisibilityChanged(bool visible);
    };

    class MyCameraInterface : public ito::AddInInterfaceBase
    {
        Q_OBJECT
    #if QT_VERSION >=  QT_VERSION_CHECK(5, 0, 0)
        Q_PLUGIN_METADATA(IID "ito.AddInInterfaceBase" )
    #endif
        Q_INTERFACES(ito::AddInInterfaceBase)
        PLUGIN_ITOM_API
        
        protected:

        public:
            MyCameraInterface();
            ~MyCameraInterface();
            ito::RetVal getAddInInst(ito::AddInBase **addInInst);

        private:
            ito::RetVal closeThisInst(ito::AddInBase **addInInst);
    };

Parameters and Unit Conventions 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to have a unified behaviour of all camera plugins, respect the following unit conventions. That means, the plugin should store related parameters using these conventions, such that **getParam** and **setParam** returns and obtains values using these units. Internally, it is sometimes
necessary to convert these units to the units required by the interface of the real camera device.

* Integration time, frame time... in **sec**
* bit depth / resolution in bit [8, 10, 12, 14, 16, 24]

Implement the following mandatory parameters in the map **m_params**:

* "name": {string | readonly}
    name of the plugin
* "bpp": {int}
    current bit depth (will be read e.g. when opening a live window)
* "sizex": {int | readonly}
    current width of the camera image (considering a possible ROI). This parameter is always read-only and needs to be changed if the optional
    parameters *x0* or *x1* change. This parameter is read e.g. when a live window is opened.
* "sizey": {int | readonly}
    current height of the camera image (considering a possible ROI). This parameter is always read-only and needs to be changed if the optional
    parameters *y0* or *y1* change. This parameter is read e.g. when a live window is opened.

If desired implement the following optional parameters in the map **m_params**:

* "integration_time": {double}
    Exposure or integration time in seconds
* "frame_time": {double}
    The time between two frames (in seconds, often read-only)
* "gain": {double}
    Normalized gain in the range [0.0,1.0]
* "offset": {double}
    Normalized offset in the range [0.0,1.0]
* "x0", "y0": {int}
    pixel coordinate of the left top corner of the image or ROI [0..width-1/height-1]
    If this changes, "sizex" or "sizey" must be changed, too.
* "x1", "y1": {int}
    pixel coordinate of the right bottom corner of the image or ROI [x0+1/y0+1..width-1/height-1]
    If this changes, "sizex" or "sizey" must be changed, too.


AD-Converters
-------------------------------------

AD-Converter plugins are directly inherited from **ito::AddInDataIO**.

.. todo::
    
    documentation for AD-converters

RawIO-Plugins
-------------------------------------

Further IO plugins are directly inherited from **ito::AddInDataIO**.

.. todo::
    
    documentation for other IO devices

