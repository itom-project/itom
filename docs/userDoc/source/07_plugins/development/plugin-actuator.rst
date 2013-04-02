.. include:: ../../include/global.inc

.. sectionauthor:: Marc Gronle

.. _plugin-class-actuator:

Plugin class - Actuator
=========================

If you want to create a plugin in order to access piezo- or motor-stages, multi-axes-machines,... it is intended to derive your plugin class from **ito::AddInActuator**. 

Base idea of any actuator
-----------------------------

The actuator interface has been developped with the following base ideas:

* The actuator can consist of different axes. The first axis is indexed with the number 0.
* The plugin contains a set of parameters (like every other plugin), which can be set or get using the public methods **setParam** or **getParam** or the approriate methods in Python.
* Some of those parameters must be available, others are optional, but if you implement them, you should follow some rules concerning name and type of the parameter, and of course you can add an infinite list of further parameters.
* Every actuator is executed in its own thread.
* Every actuator can have one configuration dialog and one docking widget toolbox, that is directly included in the GUI of |itom|.
* The current position of all axes should be stored in the vector **m_currentPos**, that is a member of class **AddInActuator**.
* The new target position for all exes must be stored in the vector **m_targetPos**, that is also a member of class **AddInActuator**.
* The current status of all axes should be stored in the vector **m_currentStatus** (member of class **AddInActuator**).
* Make sure at the initialization of your plugin that all three member vectors are initialized with the size of the numbers of axes.
* Changes to the current status or position are signaled by the signal **actuatorStatusChanged** (class **AddInActuator**).
* Changes to the target position vector is signaled by the signal **targetChanged** (class **AddInActuator**).
* Any GUI elements should only get live information about the position and status of the actuator by connecting to these signals (**actuatorStatusChanged** or **targetChanged**), since the communcation to GUI elements must be executed across multiple threads.
* Try to only connect to these signals if you really need this information, since the request of live-status and -position-information is time-consuming for certain motors. For example, a dock widget should only connect to these signals, if it is visible. This can be done by overwriting the slot **dockWidgetVisibilityChanged** of class **AddInBase**.
* Methods like **setPosAbs**, **setPosRel**, **setParam**, **calib** or **setOrigin** should only execute their given task if the motor is not moving at this moment. This can be checked using the method **isMotorMoving()**, defined in **AddInActuator**.
* The method **waitForDone** (pure virtual method of class **AddInActuator**) has to be overwritten. This method should only return if the requested axes reached their target position, reached a end switch or a timeout occurred. If the hardware is able to give sophisticated live information about the current status and position of each axis, you could continously adapt the values in the members **m_currentStatus** or **m_currentPos**; else you have to guess these values. If the axes are asynchronoulsy moved, the semaphore **waitCond** has to be released immediately before the loop waiting for the target position starts. Then the caller can directly continue working. In synchronous mode (default behaviour), **waitCond** is only released, if all axes are not moving any more.

Programming steps
-----------------

In order to program the actuator plugin, follow these steps:

#. Create the header and source file for your plugin "MyAlgoPlugin".
#. Create the interface (or factory) class "MyAlgoPluginInterface". For details about how to create such an interface class, see :ref:`plugin-interface-class`.
#. Create the plugin class "MyAlgoPlugin" with respect to the exemplary implementation, given in the next section.

    * Consider which internal parameters, that can be read and/or written by the user, your plugin has. Add these parameters in the constructor of your plugin to the **m_params**-vector.
    * Implement the **init**-method that gets the initial parameters, defined in the interface class.
    * Implement the methods **getParam** and **setParam**, which are the getter- and setter-methods for the internal parameters.
    * Implement the motor-specific methods, including **waitForDone**

Actuator plugin class
-----------------------

A sample header file of the actuator's plugin class is illustrated in the following code snippet:

.. code-block:: c++
    :linenos:
    
    #include "../../common/addInInterface.h"

    #include "dialogMyMotor.h"
    #include "dockWidgetMyMotor.h"

    class MyMotor : public ito::AddInActuator 
    {
        Q_OBJECT

        protected:
            ~MyMotor() {};	/*! < Destructor*/
            MyMotor(int uniqueID);/*! < Constructor*/


        public:
            friend class MyMotorInterface;
            const ito::RetVal showConfDialog(void);	/*!< Opens the modal configuration dialog (called from main thread) */
            int hasConfDialog(void) { return 1; }; /*!< indicates that this plugin has got a configuration dialog */

        private:
            ito::RetVal waitForDone(int timeoutMS = -1, QVector<int> axis = QVector<int>() /*if empty -> all axis*/, int flags = 0 /*for your use*/);

        signals:
            void parametersChanged(QMap<QString, ito::tParam> params);	/*!< Sends a signal if parameters have changes */

        public slots:
            //! get/set parameters
            ito::RetVal getParam(QSharedPointer<ito::tParam> val, ItomSharedSemaphore *waitCond = NULL);
            ito::RetVal setParam(QSharedPointer<ito::tParam> val, ItomSharedSemaphore *waitCond = NULL);
            
            //! init/close method
            ito::RetVal init(QVector<ito::tParam> *paramsMand, QVector<ito::tParam> *paramsOpt, ItomSharedSemaphore *waitCond = NULL); 
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

        private slots:
            void dockWidgetVisibilityChanged( bool visible ); /*!< this slot is invoked if the visibility of the dock widget has changed */
    };
    
Unit Conventions
----------------

In order to have a unique behaviour of all plugins, respect the following unit conditions:

* Length values in **mm**
* Angles in **degree**
* Velocity in **mm/sec** or **degree/sec**
* Acceleration in **mm/sec^2** or **degree/sec^2** 

Mandatory keywords for setParam / getParam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* "name": Name of the plugIn as "typeString" || "readOnly"
* "numAxis": Number of axis as "typeInt" || "readOnly"
* "async": Toggle wait behavior during movement. As "typeInt"
* "speed": Speed for ever axis of the stage as "typeDoubleArray". For single axis stages as "typeDouble". Signle axis must not throw an error of called with speed[0] 

Optional but reserved keywords for setParam / getParam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* "accel": Acceleration for ever axis of the stage as "typeDoubleArray". For single axis stages as "typeDouble". Signle axis must not throw an error of called with accel[0]
* "decel": Deceleration for ever axis of the stage as "typeDoubleArray". For single axis stages as "typeDouble". Signle axis must not throw an error of called with decel[0]  