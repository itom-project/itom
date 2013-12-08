.. include:: ../../include/global.inc

.. sectionauthor:: Alexander Bielke

.. _plugin-class-dataio:

Plugin class - DataIO
#############################

Base idea behind any DataIO
=========================================

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

The DataIO | typeGrabber plugin class
=========================================

This is a subtype of DataIO for camera / framegrabber communication. The data acquisition is managed as followed:

* The methods **startDevice** and **stopDevice** opens and closes the capture logic of the devices to reduce CDU-load. For serial ports these functions are unnecessary. 
* The method **acquire** starts the DataIO grabbing a frame with the current parameters. The function returns after sending the trigger. The function should be callable several times without calling get-/copyVal().
* The methods **getVal** and **copyVal** are the external interfaces for data grabbing. They call **retrieveData**. The function should not be callable without a previous call of **acquire** and than only once.
* In **retrieveData** the data transfer is done and frame has to copied. The function blocks until the triggered data is copied. In case retrieveData is called by getVal the frame has to be copied to **m_data**, an internal **dataObject**.
* The function **getVal** overwrites the IO-**dataObject** by a shallow copy of the internal **dataObject**. Empty objects are allowed. Warning read shallow copy of dataObject before usage.  
* The function **copyVal** deep-copies data to the external IO-**dataObject**. The **dataObject** must have the right size and type. **dataObject** with ROI should not be overwritten. The ROI should be filled. Empty objects are allowed. In case of empty **dataObject** a new object with right size is allocated.
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

	#include "../../common/addInGrabber.h"
	#include "dialogFireGrabber.h"
	#include <qsharedpointer.h>

	class FireGrabber : public ito::AddInGrabber
	{
		Q_OBJECT

		protected:
			~MyDataIO(); /*! < Destructor*/
			MyDataIO();	/*! < Constructor*/
			ito::RetVal checkData(ito::DataObject *externalDataObject = NULL);	/*!< Check if objekt has to be reallocated */
			ito::RetVal retrieveData(ito::DataObject *externalDataObject = NULL); /*!< Wait for acquired picture */


		public:
			friend class MyDataIOInterface;
			const ito::RetVal showConfDialog(void);	//! Open the config nonmodal dialog to set camera parameters 
			int hasConfDialog(void) { return 1; }; //!< indicates that this plugin has got a configuration dialog



		private:
			cv::Mat m_pDataMatBuffer;

			int m_CCD_ID; /*!< Camera ID */
			bool m_isgrabbing; /*!< Check if acquire was called */
	
			static int m_numberOfInstances;
			ito::RetVal AlliedChkError(int errornumber); /*!< Map Allied-Error-Number to ITOM-Errortype and Message */
			const char* errmsg;
			UINT8          *m_pImage;                  // Pointer to actual image

		public slots:
			//!< Get Camera-Parameter
			ito::RetVal getParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore *waitCond);
			//!< Set Camera-Parameter
			ito::RetVal setParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore *waitCond);
			//!< Initialise board, load dll, allocate buffer
			ito::RetVal init(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, ItomSharedSemaphore *waitCond = NULL);
			//!< Free buffer, delete board, unload dll
			ito::RetVal close(ItomSharedSemaphore *waitCond);

			//!< Start the camera to enable acquire-commands
			ito::RetVal startDevice(ItomSharedSemaphore *waitCond);
			//!< Stop the camera to disable acquire-commands
			ito::RetVal stopDevice(ItomSharedSemaphore *waitCond);
			//!< Softwaretrigger for the camera
			ito::RetVal acquire(const int trigger, ItomSharedSemaphore *waitCond = NULL);
			//!< Wait for acquired picture, copy the picture to dObj of right type and size
			ito::RetVal getVal(void *vpdObj, ItomSharedSemaphore *waitCond);

			ito::RetVal copyVal(void *vpdObj, ItomSharedSemaphore *waitCond);

			ito::RetVal ConvertY16(UINT32 XSize,UINT32 YSize,UINT8 *pBuf,UINT8 *pBGR);

			void updateParameters(QMap<QString, ito::Param> params);

		private slots:
		
	};

	//----------------------------------------------------------------------------------------------------------------------------------


	class DataIOInterface : public ito::AddInInterfaceBase
	{
		Q_OBJECT
			Q_INTERFACES(ito::AddInInterfaceBase)

		protected:

		public:
			DataIOInterface();
			~DataIOInterface();
			ito::RetVal getAddInInst(ito::AddInBase **addInInst);

		private:
			ito::RetVal closeThisInst(ito::AddInBase **addInInst);

		signals:

		public slots:
	};

Camera / Frame Grabber Parameters and Units
---------------------------------------------------

To achieve a unified interface and allow fast exchange between the different plugIns all dataIO have to follow the predefined conventions.

* Integrationtime named as "integration_time" in [s]
* Colordepth in bit [8,10-16,24] 

Mandatory keywords for setParam / getParam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* "name": Name of the plugIn as "typeString" || "readOnly"
* "integration_time": The integration time as "typeDouble"
* "frame_time": The time between to frames as "typeDouble" (usually "readOnly")
* "gain": The gain of the AD-Converter normed between 0..1 as "typeDouble"
* "offset": The offset of the AD-Converter normed between 0..1 as "typeDouble"
* "x0": first pixel of ROI as "typeInt"
* "x1": last pixel of ROI as "typeInt"
* "y0": first pixel of ROI as "typeInt"
* "y1": last pixel of ROI as "typeInt"
* "sizex": Current width of the grabber-ROI as "typeInt" || "readOnly"
* "sizey": Current heigth of the grabber-ROI as "typeInt" || "readOnly"
* "bpp": Current bits per pixel of the grabber as "typeInt". Usual values are 8-bit, 10-bit - 16-bit, 24-bit color. 

Optional but reserved keywords for setParam / getParam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* "binning": Binning of the pixels in both directions as "typeInt" in form [BinningX? BinningY?] between "0101" and "xy" 


The DataIO | typeADDA plugin class
=========================================



The DataIO | typeRawIO plugin class
=========================================

