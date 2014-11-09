.. include:: ../../include/global.inc

.. sectionauthor:: Alexander Bielke

.. _plugin-class-dataio:

Plugin class - DataIO
=========================

Base idea behind any DataIO
-----------------------------

* Plugins of class **dataIO** operate systems that have input or output data, e.g. frame grabbers, cameras, serial ports, AD-DA-converters...
* Every single instance of class *dataIO* has an exclusive communication with one connected device. More cameras or ADDA-converter of the same type should be managed with a corresponding number of instances.
* Depending on the plugin, more than one DataIO can be separately controlled by the plug-number or the vendor/camera-ID.
* Like any other plugin, every DataIO has a set of parameters, which can be set/get using python by the commands **setParam** and **getParam**.
* Every DataIO is executed in its own thread.
* Every DataIO can have one configuration dialog and one toolbox (dockWidget).
* All parameters are stored in the member **m_params** of type QMap. They can be read or changed by methods **getParam** and **setParam**. Some parameters are read only!
* If parameters of DataIO change, they also must be updated in the **m_params**-map and the signal **parametersChanged** must be emitted.
* The data acquisition is performed according to the grabber subtype. These subtypes are 'typeGrabber', 'typeADDA' and 'typeRawIO'.

Grabber plugin
------------------------------

This is a subtype of DataIO for camera / framegrabber communication. Plugins of this type are inherited from **ito::AddInGrabber**. The data acquisition is managed as follows:

* The methods **startDevice** and **stopDevice** open and close the capture logic of the devices to reduce CPU-load. For serial ports these functions are unnecessary. 
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

    
A sample header file of the DataIO's plugin class might look like the following snippet:

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
* "x0", "y0": {int, deprecated (see roi)}
    pixel coordinate of the left top corner of the image or ROI [0..width-1/height-1]
    If this changes, "sizex" or "sizey" must be changed, too.
* "x1", "y1": {int, deprecated (see roi)}
    pixel coordinate of the right bottom corner of the image or ROI [x0+1/y0+1..width-1/height-1]
    If this changes, "sizex" or "sizey" must be changed, too.
* "roi": {int-array}
    Since itom AddIn Interface version 1.3.1 (itom 1.4.0 or higher), it is recommended to replace *x0* *y0*, *x1* and *y1* by the integer array based
    parameter **roi** which expects an array [left, top, width, height]. This parameter can easily be parametrized using the meta information ito::RectMeta
    and allows the direct configuration of the entire ROI or a single access to one of the four components, by passing the parametername *roi[0]*, *roi[1]*....



AD-Converters
-------------------------------------

AD-Converter plugins are directly inherited from **ito::AddInDataIO**. An AD-DA-converter plugin has the following characteristics:

* It can communicate with 1 or multiple input and/or output channels. To set the number of total channels and to define if a channel is an incoming or outgoing channel, the plugin's parameters or initialization parameters should be used. Sometimes it is not possible to change the direction after initialization, this depends on the device.
* The method **startDevice** must be called like in a camera before the first usage of the device in order to establish the connection. Create a no-operation method, if this is not necessary for your device. It is possible, that startDevice is called multiple time, therefore count the number of starts and only establish the connection upon the first call.
* As counterpart to **startDevice**, the method **stopDevice** disconnects the device. For every call of **startDevice**, **stopDevice** must be called and at the last call, the connection should be interrupted.
* In difference to a camera dataIO plugin, the method **acquire** can be used to start the acquisition of a serie of input data values at all previously selected input channels. It is also possible to create an empty function here. Then the reading-process of new single data values for each input channel is totally executed in the method **getVal**.
* The method **copyVal** needs not to be implemented for AD-DA-converters.
* Method **getVal**: This method registers input values from all previously selected input channels (depending on your implementation and parametrization it is also possible to register multiple values per channel) and returns these values to the user. 

    * If you have one or multiple input channels, use the definition **getVal(void \*dObj, ItomSharedSemaphore \*waitCond)**. The parameter dObj is then a pointer to ito::DataObject and can be cast to this class. Return an MxN data object, where M corresponds to the number of read input channels and N corresponds to the data samples per channel. If you want to, you can also force the user to previously allocate the given data object such that you can get a hint how many samples should be registered.
    * For only one input channel, it is also possible to implement the definition **getVal(QSharedPointer<char> data, QSharedPointer<int> length, ItomSharedSemaphore *waitCond = NULL)** where an allocated char-buffer whose size is defined by *length* is given. Fill in the data samples into the buffer (considering the given length) or use the length value to see how many samples are requested.
    
* Method **setVal**: This method is called if the user wants the plugin to set data to all selected output channels. The definition is **setVal(const char \*data, const int length, ItomSharedSemaphore \*waitCond = NULL)**. In case of AD-DA-converter plugins, length is always 1 and *data* must be cast to **ito::DataObject\***. The given data object must then have a size of MxN where M denotes the number of output channels (must correspond to the number of channels to write data) and N is the number of samples. You can then send all samples to each channel either as fast as possible or using a timer, using the timer of the device. This depends on its abilities.

A sample header file of the DataIO's plugin class for AD-converters might look like the following snippet:

.. code-block:: c++
    :linenos:

    #include "common/addInInterface.h"
    #include <qsharedpointer.h>

    class MyADConverter : public ito::AddInDataIO
    {
        Q_OBJECT

        protected:
            ~MyADConverter(); /*! < Destructor*/
            MyADConverter();    /*! < Constructor*/

        public:
            friend class MyADConverterInterface;
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

            ito::RetVal getVal(void *vpdObj, ItomSharedSemaphore *waitCond);
            ito::RetVal getVal(QSharedPointer<char> data, QSharedPointer<int> length, ItomSharedSemaphore *waitCond = NULL);
            ito::RetVal setVal(const char *data, const int length, ItomSharedSemaphore *waitCond = NULL);

        private slots:
            void dockWidgetVisibilityChanged(bool visible);
    };

    class MyADConverterInterface : public ito::AddInInterfaceBase
    {
        Q_OBJECT
    #if QT_VERSION >=  QT_VERSION_CHECK(5, 0, 0)
        Q_PLUGIN_METADATA(IID "ito.AddInInterfaceBase" )
    #endif
        Q_INTERFACES(ito::AddInInterfaceBase)
        PLUGIN_ITOM_API
        
        protected:

        public:
            MyADConverterInterface();
            ~MyADConverterInterface();
            ito::RetVal getAddInInst(ito::AddInBase **addInInst);

        private:
            ito::RetVal closeThisInst(ito::AddInBase **addInInst);
    };

RawIO-Plugins
-------------------------------------

Further IO plugins are directly inherited from **ito::AddInDataIO**.

.. todo::
    
    documentation for other IO devices

