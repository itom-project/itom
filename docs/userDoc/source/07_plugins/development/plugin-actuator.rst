.. include:: ../../include/global.inc

.. sectionauthor:: Marc Gronle

.. _plugin-class-actuator:

Plugin class - Actuator
=========================

If you want to create a plugin in order to access piezo- or motor-stages, multi-axes-machines,... it is intended to derive your plugin class from **ito::AddInActuator**. 

Base idea of any actuator
-----------------------------

The actuator interface has been developed with the following base ideas:

* The actuator can consist of different axes. The first axis is indexed with the number 0. Create the read-only, integer parameter **numAxis** for the number of connected axes.

* The plugin contains a set of parameters (like every other plugin), which can be set or get using the public methods **setParam** or **getParam** or the appropriate methods in Python.

* emit the signal **parametersChanged(m_params)** if any parameter has been changed in order to inform the GUI about these changes.

* Some of those parameters must be available, others are optional, but if you implement them, you should follow some rules concerning name and type of the parameter, and of course you can add an infinite list of further parameters (see below).

* Every actuator is executed in its own thread.

* Every actuator can have one configuration dialog and one dockable toolbox, that is directly included in the GUI of |itom|.

* The current position of all axes should be stored in the vector **m_currentPos**, that is a member of class **AddInActuator** (bitmask of enumeration **ito::tActuatorStatus**.

* The new target position for all axes must be stored in the vector **m_targetPos**, that is also a member of class **AddInActuator**.

* The current status of all axes should be stored in the vector **m_currentStatus** (member of class **AddInActuator**).

* Make sure at the initialization of your plugin that all three member vectors are initialized with the size of the numbers of axes.

* Changes to the current status or position are signalled by the signal **actuatorStatusChanged** (class **AddInActuator**). This is usually emitted by calling **sendStatusUpdate(...)**.

* Changes to the target position vector is signalled by the signal **targetChanged** (class **AddInActuator**). This is usually emitted by calling **sendTargetUpdate()**. Connected toolboxes as well as further GUI elements are then informed about the changes.

* Any GUI elements should only get live information about the position and status of the actuator by connecting to these signals (**actuatorStatusChanged** or **targetChanged**), since the communication to GUI elements must be executed across multiple threads.

* Try to only connect to these signals if you really need this information, since the request of live-status and -position-information is time-consuming for certain motors. For example, a dock widget should only connect to these signals, if it is visible. This can be done by overwriting the slot **dockWidgetVisibilityChanged** of class **AddInBase**.

* Methods like **setPosAbs**, **setPosRel**, **setParam**, **calib** or **setOrigin** should only execute their given task if the motor is not moving at this moment. This can be checked using the method **isMotorMoving()**, defined in **AddInActuator** by simply checking the appropriate status flags.

* The method **waitForDone** (pure virtual method of class **AddInActuator**) **has to be overwritten**. This method continuously checks the moving status of all (moving) axes are returns if all requested axes reached their target position, reached a switch or if a time-out occurred. User interrupts are also checked within this function. In case of such an interrupt the axes status are set to *interrupt* as well and the method returns. If the hardware is able to give sophisticated live information about the current status and position of each axis, you can continuously adapt the values in the members **m_currentStatus** or **m_currentPos**; else you can to guess these values. Signal changes to these vectors using the methods **sendStatusUpdate()** or **sendStatusUpdate(...)**. If the axes are asynchronously moved, the semaphore **waitCond** has to be released immediately before the loop waiting for the target position starts. Then the caller can directly continue working. In synchronous mode (default behaviour), **waitCond** is only released, if no requested axis is moving any more.

* There is a slot **requestStatusAndPosition(bool sendActPosition, bool sendTargetPos)**, defined in **ito::AddInActuator**. This slot is invoked e.g. by a toolbox or configuration dialog in order to force the actuator to directly emit the signals **actuatorStatusChanged** and/or **targetChanged**. The original caller is then immediately informed about the current status and position values. Overload this function if you want to update **m_currentStatus**, **m_currentPos** or **m_targetPos** before they are emitted to the caller. In the default implementation, they are emitted as they are.

Programming steps
-----------------

In order to program the actuator plugin, follow these steps:

#. Create the header and source file for your plugin "MyActuatorPlugin".
#. Create the interface (or factory) class "MyActuatorPluginInterface". For details about how to create such an interface class, see :ref:`plugin-interface-class`.
#. Create the plugin class "MyActuatorPlugin" with respect to the exemplary implementation, given in the next section.

    * Consider which internal parameters, that can be read and/or written by the user, your plugin has. Add these parameters in the constructor of your plugin to the **m_params**-vector.
    * Implement the **init**-method that gets the initial parameters, defined in the interface class.
    * Implement the methods **getParam** and **setParam**, which are the getter- and setter-methods for the internal parameters.
    * Implement the motor-specific methods, including **waitForDone**

Actuator plugin class
-----------------------

A sample header file of the actuator's plugin class is illustrated in the following code snippet:

.. code-block:: c++
    :linenos:
    
    #define ITOM_IMPORT_API
    #define ITOM_IMPORT_PLOTAPI

    #include "../../common/addInInterface.h"

    #include "dialogMyMotor.h"
    #include "dockWidgetMyMotor.h"

    class MyMotor : public ito::AddInActuator 
    {
        Q_OBJECT

        protected:
            ~MyMotor() {};	/*! < Destructor*/
            MyMotor();/*! < Constructor*/
            
            ito::RetVal waitForDone(int timeoutMS = -1, QVector<int> axis = QVector<int>() /*if empty -> all axis*/, int flags = 0 /*for your use*/);
            
        public:
            friend class MyMotorInterface;
            const ito::RetVal showConfDialog(void);	/*!< Opens the modal configuration dialog (called from main thread) */
            int hasConfDialog(void) { return 1; }; /*!< indicates that this plugin has got a configuration dialog */

        public slots:
            //! get/set parameters
            ito::RetVal getParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore *waitCond = NULL);
            ito::RetVal setParam(QSharedPointer<ito::Param> val, ItomSharedSemaphore *waitCond = NULL);
            
            //! init/close method
            ito::RetVal init(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, ItomSharedSemaphore *waitCond = NULL); 
            ito::RetVal close(ItomSharedSemaphore *waitCond);

            //! calibration for single or multiple axis
            ito::RetVal calib(const int axis, ItomSharedSemaphore *waitCond = NULL);
            ito::RetVal calib(const QVector<int> axis, ItomSharedSemaphore *waitCond = NULL);
            
            //! current axis position is new zero-position
            ito::RetVal setOrigin(const int axis, ItomSharedSemaphore *waitCond = NULL);
            ito::RetVal setOrigin(const QVector<int> axis, ItomSharedSemaphore *waitCond = NULL);
            
            //! Reads out status request answer and gives back ito::retOk or ito::retError
            ito::RetVal getStatus(QSharedPointer<QVector<int> > status, ItomSharedSemaphore *waitCond);

            //! get current position of single or multiple axis (in mm or degree)
            ito::RetVal getPos(const int axis, QSharedPointer<double> pos, ItomSharedSemaphore *waitCond);
            ito::RetVal getPos(const QVector<int> axis, QSharedPointer<QVector<double> > pos, ItomSharedSemaphore *waitCond);
            
            //! move one or more axis to certain absolute positions (in mm or degree)
            ito::RetVal setPosAbs(const int axis, const double pos, ItomSharedSemaphore *waitCond = NULL);
            ito::RetVal setPosAbs(const QVector<int> axis, QVector<double> pos, ItomSharedSemaphore *waitCond = NULL);
            
            //! move one or more axis by certain relative distances (in mm or degree)
            ito::RetVal setPosRel(const int axis, const double pos, ItomSharedSemaphore *waitCond = NULL);
            ito::RetVal setPosRel(const QVector<int> axis, QVector<double> pos, ItomSharedSemaphore *waitCond = NULL);
            
            //! if this slot is triggered, the current status and position is emitted (e.g. for actualizing a dock widget)
            ito::RetVal RequestStatusAndPosition(bool sendActPosition, bool sendTargetPos);
            
            //ito::RetVal requestStatusAndPosition(bool sendCurrentPos, bool sendTargetPos); //!see notes above

        private slots:
            void dockWidgetVisibilityChanged( bool visible ); /*!< this slot is invoked if the visibility of the dock widget has changed */
    };

The corresponding source file should start with something like this:

.. code-block:: c++
    :linenos:
    
    #define ITOM_IMPORT_API
    #define ITOM_IMPORT_PLOTAPI
    
    #include "yourHeaderFile.h"
    
    //implement your code here
    
Signalling the current position and status of any axes
-------------------------------------------------------

Each actuator has the possibility to signalize the target position, the current position and the current status of each axis. Then its own toolbox or other widgets or slots (general: listeners) can be connected to the corresponding signals in order to be informed about the current activity. The base class **ito::AddInActuator** provides the necessary structures for this:

1. The vector **m_currentPos** must be initialized to a length corresponding to the number of axes and contains the current position of every axis using the units stated below. Whenever
the actuator registers a change of any current position, the corresponding value should be changed as well. Listeners are finally informed about this change by calling the method 

    .. code-block:: c++
        
        sendStatusUpdate(false)

    The argument **false** means that not only a change of the current status happened, but also a change of any current position. This method internally emits the signal **actuatorStatusChanged**.

2. The vector **m_targetPos** must also be initialized to a length corresponding to the number of axes. Whenever a positioning operation starts, set the target value of specific axes to the new target value and call

    .. code-block:: c++
        
        sendTargetUpdate()

    that finally emits the signal **targetChanged**.

3. The status of every axis is stored in the vector **m_currentStatus**. Each item in this vector with a length corresponding to the number of axes, contains an OR combination of the enumeration **ito::tActuatorStatus**. Whenever the status of any axis changes, change its status value, too and use **sendStatusUpdate(true/false)** in order to also emit the signal **actuatorStatusChanged**.

The enumeration **ito::tActuatorStatus** contains the following values that are grouped by specific mask values:

The **moving flags** contain flags about the current moving status of any axis (bits containing to this group are contained in the mask **ito::actMovingMask**):

    * ito::actuatorUnknown: The current status of this axis is unknown
    * ito::actuatorInterrupted: The movement of this axis has been interrupted and no further commands followed
    * ito::actuatorMoving: The axis is currently moving (or is supposed to move)
    * ito::actuatorAtTarget: The axis reached its target position (this is the default value)
    * ito::actuatorTimeout: A timeout occurred during the movement of this axis

The **status flags** inform about the general status of any axis (bits containing to this group are set in the mask **ito::actStatusMask**):
    
    * ito::actuatorAvailable: This axis is available (usually set)
    * ito::actuatorEnabled: This axis is enabled and can be driven (usually set, but there are drivers that allowing disabling selected axis)

Axes that have got any reference or end switches can signal related status information using the **switches flags**. All bits belonging to this group are set in the mask **ito::actSwitchesMask** divided into **ito::actEndSwitchMask** and **ito::actRefSwitchMask**):
    
    * ito::actuatorEndSwitch: This bit is set if any (unknown) end switch was reached
    * ito::actuatorLeftEndSwitch: This bit is additionally set if the left end switch was reached
    * ito::actuatorRightEndSwitch: This bit is additionally set if the right end switch was reached
    * ito::actuatorRefSwitch: This bit is set if any (unknown) reference switch was reached
    * ito::actuatorLeftRefSwitch: This bit is additionally set if the left reference switch was reached
    * ito::actuatorRightRefSwitch: This bit is additionally set if the right reference switch was reached

You can either manually set the necessary bit-combination of moving, status and switch flags for signalling the right status of the axis. There are three methods defined in **ito::AddInActuator** that simplify this process:

.. code-block:: c++
    
    setStatus(int &status, const int newFlags, const int keepMask = 0)
    setStatus(const QVector<int> &axis, const int newFlags, const in keepMask = 0)

Use this methods to the set the status of one or multiple axis. The parameter **newFlags** should contain an or-combination of all flags that should be set. The status flags are then set to this value (hence, old values are overwritten). If you want to keep the current bit values of a certain group, pass the specific mask as argument **keepMask**. For instance, if you want to the status of the second axis to **actuatorMoving** without changing the **status flags**, use the following command:

.. code-block:: c++
    
    setStatus(m_currentStatus[1], ito::actuatorMoving, ito::actStatusFlags)
    # this command will set all bits of the switches mask to 0!

The equivalent command for multiple axis, requires a vector with axes-indices as first argument. This example does the same for the first and third axis:

.. code-block:: c++
    
    QVector<int> axis;
    axis << 0 << 2;
    setStatus(axis, ito::actuatorMoving, ito::actStatusFlags)

The similar commands **replaceStatus**

.. code-block:: c++
    
    replaceStatus(int &status, const int existingFlag, const int replaceFlag)
    replaceStatus(const QVector<int> &axis, const int existingFlag, const int replaceFlag)

can be used to replace one status flag by another one without changing the other bits. If the bit corresponding to the **existingFlag** is set, it is set to zero and the bit of the **replaceFlag** is set to 1. In the following example, the flag of the first axis is set from moving to atTarget:

.. code-block:: c++
    
    replaceStatus(m_currentStatus[0], ito::actuatorMoving, ito::actuatorAtTarget)

After using one of these functions to set the current status, call **sendStatusUpdate** to emit the signal **actuatorStatusChanged** such that connected listeners can for instance visualize the current status.

Interruption of movement
--------------------------

It is possible to implement an interrupt button in the toolbox of the actuator that becomes active once at least one axis is moving. Once the button is clicked it must **directly** call the thread-safe function **setInterrupt()** of the actuator plugin (If the toolbox inherits from **ito::AbstractAddInDockWidget** call its method **setActuatorInterrupt()**.

In the method **waitForDone** regularly check if the interrupt flag has been set, using the actuator's method **isInterrupted()**. If this method
returns true, set the moving state of all moving axes to **ito::actuatorInterrupted** and return with an appropriate return value, like:

.. code-block:: c++
    
    return ito::RetVal(ito::retError, 0, "movement interrupted");

.. note::
    
    Once **isInterrupted()** returns true, the internal interrupt flag is reset to false. Therefore consider to call this function to reset the interrupt flag if desired (e.g. at the begin of the next movement).


    
Parameters and Unit Conventions 
---------------------------------

In order to have a unified behaviour of all actuator plugins, respect the following unit conventions. That means, the plugin should store related parameters using these conventions, such that **getParam** and **setParam** returns and obtains values using these units. Internally, it is sometimes
necessary to convert these units to the units required by the interface of the real actuator device.

* Length values in **mm**
* Angles in **degree**
* Velocity in **mm/sec** or **degree/sec**
* Acceleration, deceleration in **mm/sec^2** or **degree/sec^2** 

Implement the following mandatory parameters in the map **m_params**:

* "name": {string | readonly}
    name of the plugin
* "numaxis": {int | readonly}
    number of connected axes
* "async": {int, [0,1]}
    If 1: asynchronous movement. Methods like **setPosAbs** or **setPosRel** only start the movement and immediately return. Hence, the *waitCond* in **waitForDone** is directly released before the loop waiting for the end of the movement is executed. If 0: synchronous movement (default). **setPosAbs** and **setPosRel** block until the end of the movement, hence, *waitCond* in **waitForDone** is only released at the end of the movement. Since **waitForDone** always is running during the movement, the plugin thread is blocked and no further commands can be executed, even in asynchronous mode.

If desired implement the following optional parameters in the map **m_params**:

* "speed": {double or doubleArray}
    Desired speed for the axes. If *double*, the speed holds for all axes, else the *doubleArray* must have the same length than the number of axes, holding the axis specific speed values. Make sure, that it is not possible to set an array of another length in **setParam**.
* "accel": {double or doubleArray}
    Acceleration values (similar to *speed*)
* "decel": {double or doubleArray}
    Deceleration values (similar to *speed*)