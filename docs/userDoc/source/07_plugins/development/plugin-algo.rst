.. include:: ../../include/global.inc

.. sectionauthor:: Florian Mauch, Marc Gronle, ITO

.. _plugin-class-algo:

Plugin class - Algo
===================

An **algorithm plugin** can provide an arbitrary number of filter-methods and widgets, hence external windows, dialogs... which can be displayed by |itom|.

**Filter-Method**

A filter-method is any algorithm, which is parameterized with a set of mandatory and optional parameters and may return a set of output parameters. Similar
to the flexible initialization of plugins of type **dataIO** or **actuator**, the default-values of these parameters are given by the plugin itself and can also
be different for each filter-method. Please consider that a filter-method can finally be called from different threads, therefore it is not allowed to directly
use any GUI-elements in such plugins. If the appropriate method is called by the python function **filter**, it is executed in the python thread, if the method
is called form any toolbox or dialog, it is usually called in their context (main thread) but it can also be, that the filter-method is executed by any other
worker thread. If a filter-method only consists of ordinary algorithmic components, you should not get any problems with these kind of flexibility.

**Widgets**

The widget-methods of the plugin will be called in the same way than filter-methods with a set of mandatory and optional parameters. Then, the widget-method creates
a new instance of a widget, window or any other GUI element, derived from **QWidget** is returned and can then be displayed. This method will always be called in the
context of the main thread (GUI-thread).

For the usage of such filters via Python, see :ref:`getStartFilter`.

Plugin-Structure
----------------

Like all the other types of plugins, the **Algo**-plugin must at least consist of two classes. The first (*here:* **MyAlgoPluginInterface**) is the interface, or factory
class, necessary for the successful load of the plugin by the |Qt|-plugin system. The "real" plugin class (*here:* **MyAlgoPlugin**) is then accessed by |itom| through
appropriate methods in class **MyAlgoPluginInterface**.

Factory-Class
-------------

The structure and exemplary implementation of a factory class **MyAlgoPluginInterface** is mainly explained in :ref:`plugin-interface-class`. Since the main class **MyAlgoPlugin**
is mainly behaving like a static, singleton class, it will never be instantiated by the user, like it is the case for plugins of type **dataIO** or **actuator**. Therefore
it makes no sense to define some default mandatory and optional parameters for the constructor of **MyAlgoPlugin**, such that the programmer should the vectors **m_initParamsMand**
and **m_initParamsOpt** be unchanged.

Plugin-Class
------------

The raw scheme for your plugin-class is as follows:

**Header-File (myPluginAlgo.h)**

.. code-block:: c++
    :linenos:

    #ifndef MYALGOPLUGIN_H
    #define MYALGOPLUGIN_H

    #include "../../common/addInInterface.h" //or similar path

    class MyPluginAlgoInterface : public ito::AddInInterfaceBase
    {
    ... //see in documentation for interface-class
    };

    class MyPluginAlgo : public ito::AddInAlgo
    {
        Q_OBJECT

    protected:
        MyPluginAlgo(int uniqueID);
        ~MyPluginAlgo() {};

    public:
        friend class MyAlgoPluginInterface;

        //filter-method1
        static ito::RetVal filter1(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut);
        static ito::RetVal filter1Params(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut);

        //for every further filter-method define another pair like above

        //widget-method1
        static QWidget* widget1(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ito::RetVal &retValue);
        static ito::RetVal widget1Params(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut);

        //for every further widget-method define another pair like above

    public slots:
        ito::RetVal init(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ItomSharedSemaphore *waitCond = NULL);
        ito::RetVal close(ItomSharedSemaphore *waitCond);
    };

    #endif

**Source File (myPluginAlgo.cpp)**

.. code-block:: c++
    :linenos:

    #define ITOM_IMPORT_API
    #define ITOM_IMPORT_PLOTAPI

    #include "myPluginAlgo.h"

    //implement your code here

First of all, our algorithm plugin class is derived from the class **AddInAlgo** from within the **ito**-namespace. This base class
is defined in the *addInInterface.h* header that has to be included. Again, our plugin is ultimately derived from **QObject** and
the *Q_OBJECT* macro must appear in the class definition in order to be able to use any services provided by Qt's meta-object
system, such as the signal slot mechanisms. Additionally our plugin class has to be a friend of its interface class (see section
:ref:`plugin-interface-class`), such that the factory (interface) class is allowed to access the protected constructor.

Every filter method as well as widget method consist of two different methods. One is the real algorithm or function itself
(*here:* named with *filter1* and *widget1*). The second method is a small parameter-method which generates the default vectors for the mandatory, optional
and output (filters only) parameters. This method is necessary, since the |itom| base application and the python scripts have no
way of knowing what parameters the filter methods or widget generation methods expect.

These default-parameter methods have the following implementation:

.. code-block:: c++
    :linenos:

    ito::RetVal MyAlgoPlugin::filter1Params(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut)
    {
        ito::Param param;
        ito::RetVal retval = ito::retOk;
        retval += prepareParamVectors(paramsMand,paramsOpt,paramsOut);
        if(retval.containsError()) return retval;

        param = ito::Param("mand1", ito::ParamBase::DObjPtr | ito::ParamBase::In, NULL, tr("description").toLatin1().data());
        paramsMand->append(param);
        param = ito::Param("mand2", ito::ParamBase::String | ito::ParamBase::In, NULL, tr("description").toLatin1().data());
        paramsMand->append(param);
        param = ito::Param("opt1",ito::ParamBase::Double | ito::ParamBase::In, 0.0, 1.0, 0.0, tr("description").toLatin1().data());
        paramsOpt->append(param);

        param = ito::Param("return1", ito::ParamBase::Int | ito::ParamBase::Out, NULL, tr("description").toLatin1().data());
        paramsOut->append(param);
        param = ito::Param("return2", ito::ParamBase::String | ito::ParamBase::Out, NULL, tr("description").toLatin1().data());
        paramsOut->append(param);

        return retval;
    }

In this exemplary case, the method **filter1** requires two mandatory parameters, where the first one is a dataObject-pointer and the second one is a string.
Additionally one can give one optional parameter, namely a double-value. This filter does not have output-parameters.

.. note::
    Please don't forget to specify your parameters with the flags **In**, **Out** or **In|Out**. Mandatory and optional parameters can not be **Out** only,
    however output parameters must be **Out** only. Please consider that pure **Out** values is only allowed for non-pointer-type values (hence: Char, Int,
    Double, String, IntArray, CharArray and DoubleArray are allowed). For widget-methods, the output-value default implementation is ignored and therefore the
    corresponding vector should not be changed.

In line 3 of the method above, the three arguments are checked for not being NULL and are cleared within the method **prepareParamVectors**, which is already
defined in class **ito::AddInAlgo**. Next, the default implementations of parameters (see :ref:`plugin-Params`) are created and appended to the corresponding
vector. Finally the method is called in order to get the three default vectors. Then, the values are merged with the values given by the user or other
plugins and the real filter- or widget-method is called with the same vectors, however each parameter is now casted from **Param** to **ParamBase**.

.. note::
    In order to have an efficient call, the parameter method is only called once at startup of |itom| and the default implementations are hashed by **ito::AddInManager**. The
    consequence for the plugin programmer is, that it is not allowed to have time- or situation-dependent changes in the default-values. This might not be considered.
    Only the implementation at startup is relevant and must not be changed!

The *plugin*-methods give the user additionally the possibility to get a readable output of the set of desired
parameters including their descriptions and types in the command line by using the following python-command:

.. code-block:: python

    filterHelp("MyAlgoPlugin")

If the argument string does not fit to any specific filter, an enumeration of all filters containing this string will be printed.

Now the filter-method or widget-method itself can be implemented. While the widget-method only provide one method definition,
the filter method can have two different possible definitions (since itom 3.3).:

Filter-Methods (Without status information and / or cancellation feature)
---------------------------------------------------------------------------

This is the default definition of filters and does not provide the possibility to pass runtime status information (like the current progress)
or a way to let the user cancel the execution of the algorithm.

After that you implemented the parameter-method in order to generate default parameters for your filter-method
(see section above), you can now implement the filter-method itself. This first implementation might follow this scheme:

.. code-block:: c++
    :linenos:

    ito::RetVal MyAlgoPlugin::filter1(QVector<ito::ParamBase> *paramsMand,
                                      QVector<ito::ParamBase> *paramsOpt,
                                      QVector<ito::ParamBase> *paramsOut)
    {
        ito::RetVal retval = ito::retOk;

        //1. Section. Getting typed in or in/out parameters from paramsMand and paramsOpt
        //  Make sure that you access only parameters, that have been defined in the corresponding parameter-method.
        //  The order and type is important.

        //possibility 1 (index-based access):
        const ito::DataObject *dObj = (*paramsMand)[0].getVal<const ito::DataObject*>();
        const char *filename = (*paramsMand)[1].getVal<char*>(); //don't delete this pointer (borrowed)
        double opt1 = (*paramsOpt)[0].getVal<double>();

        //possibility 2 (name-based access):
        const ito::DataObject *dObj2 = \
            (const ito::DataObject*)ito::getParamByName(paramsMand, "mand1", &retval)->getVal<void*>();
        const char *filename2 = ito::getParamByName(paramsMand, "mand2", &retval)->getVal<char*>();
        double opt2 = ito::getParamByName(paramsOpt, "opt1", &retval)->getVal<double>();

        //2. Section. Algorithm.
        //  include here your algorithm. make sure, that you only change values of pointer-based parameters
        //  that you have defined with the flags In|Out.

        //3. Section. Optionally put results into the paramsOut-vector
        //  Make sure that you defined the corresponding parameter with right type in the corresponding parameter-method.
        //  The implementation in the next lines is only one example and has not be defined before. Be aware of that.
        (*paramsOut)[0].setVal<int>(2);
        (*paramsOut)[1].setVal<char*>("we are done");

        return retval;
    }

For the parsing of the given input parameters, you can (like stated above) either directly access them using their index in the vector or you
can use the high-level method **getParamByName**, which is provided in the file **helperCommon.h** and **helperCommon.cpp** in the **common**-folder.
If you want to use this method, integrate both files in your project and include the header file in your plugin file.

.. note::

    Always make sure, that if you are access (read or write) any parameter in any of the three vectors, you must have the specific parameter
    defined in the corresponding parameter method. If you defined there a parameter of certain type and appended it to one of the vectors, you
    can be sure, that a parameter of same type is available at the same position in the arguments of the filter-method call.


.. note::

    If you want to use methods provided by the |itom|-API, see :ref:`plugin-itomAPI` and consider the additional lines of code in your implementation.

.. note::

    Never use the reserved name **_observer** for a mandatory or optional parameter name, since this is used to call a filter
    from the Python method :py:meth:`itom.filter` with a given progress observer (see :py:class:`itom.progressObserver`).

.. _plugin-class-algo-filterDefExt:

Filter-Methods 2 (With status information and / or cancellation feature)
---------------------------------------------------------------------------

This 2nd definition of a filter method is introduced with itom 3.3 on. Its advantage is, that another argument
**QSharedPointer<ito::FunctionCancellationAndObserver> observer** is passed to the filter. The class **ito::FunctionCancellationAndObserver**
is used to provide mechanisms, such that the filter algorithm can continuously report its current progress. Additionally, this class
has an interrupt flag, which can be set via itom (e.g. via Python, if a Python script is interrupted or from a calling C++ method).
It is the responsibility of the filter developer to regularly check the state of this interrupt flag. Once it is set, try to stop
the algorithm as soon as possible and return with an error (ito::retError) set.

The filter definition for this 2nd case is as follows:

.. code-block:: c++
    :linenos:

    ito::RetVal MyAlgoPlugin::filter2(QVector<ito::ParamBase> *paramsMand,
                                      QVector<ito::ParamBase> *paramsOpt,
                                      QVector<ito::ParamBase> *paramsOut,
                                      QSharedPointer<ito::FunctionCancellationAndObserver> observer)
    {
        ito::RetVal retval = ito::retOk;

        //indicate the start of the algorithm to the observer (if given)
        if (observer)
        {
            observer->setProgressValue(observer->progressMinimum());
            observer->setProgressText("Start of algorithm");
        }

        //1. Section. Getting typed in or in/out parameters from paramsMand and paramsOpt
        //  Make sure that you access only parameters, that have been defined in
        //  the corresponding parameter-method. The order and type is important.

        //possibility 1 (index-based access):
        const ito::DataObject *dObj = (*paramsMand)[0].getVal<const ito::DataObject*>();
        const char *filename = (*paramsMand)[1].getVal<char*>(); //don't delete this pointer (borrowed)
        double opt1 = (*paramsOpt)[0].getVal<double>();

        //possibility 2 (name-based access):
        const ito::DataObject *dObj2 = \
            (const ito::DataObject*)ito::getParamByName(paramsMand, "mand1", &retval)->getVal<void*>();
        const char *filename2 = ito::getParamByName(paramsMand, "mand2", &retval)->getVal<char*>();
        double opt2 = ito::getParamByName(paramsOpt, "opt1", &retval)->getVal<double>();

        //2. Section. Algorithm.
        //  The following algorithm needs around 10seconds to finish. It reports its progress every second and
        //  if the interrupt flag of the observer is set, it quits the execution earlier:
        QElapsedTimer timer;
        timer.start();
        qint64 nextProgressReport = 1000; //every second

        while (timer.elapsed() < 10000)
        {
            if (observer)
            {
                if (timer.elapsed() >= nextProgressReport)
                {
                    //always pass the value between the given minimum / maximum of the observer
                    int value = observer->progressMinimum() + timer.elapsed() * \
                        (observer->progressMaximum() - observer->progressMinimum()) / 10000;
                    observer->setProgressValue(value);
                    observer->setProgressText(QString("This algorithm run %1 from 10.0 seconds"). \
                        arg(timer.elapsed() / 1000));
                    nextProgressReport += 1000;
                }

                if (observer->isCancelled())
                {
                    retval += ito::RetVal(ito::retError, 0, "algorithm cancelled");
                    break;
                }
            }

            QThread::msleep(100);

        }
        //  that you have defined with the flags In|Out.

        //3. Section. Optionally put results into the paramsOut-vector
        //  Make sure that you defined the corresponding parameter with right type in the corresponding parameter-method.
        //  The implementation in the next lines is only one example and has not be defined before. Be aware of that.
        (*paramsOut)[0].setVal<int>(2);
        (*paramsOut)[1].setVal<char*>("we are done");

        return retval;
    }

Widget-Method (GUI-Extensions)
------------------------------

If you want to provide a user-defined window, dialog or widget (which is then rendered into a dialog), you have to
implement an appropriate method which follows this base structure:

.. code-block:: c++
    :linenos:

    QWidget* MyAlgoPlugin::widget1(QVector<ito::ParamBase> *paramsMand,
                                   QVector<ito::ParamBase> *paramsOpt,
                                   ito::RetVal &retValue)
    {
        //1. Section. Getting typed in or in/out parameters from paramsMand and paramsOpt
        //   Make sure that you access only parameters, that have been defined in
        //   the corresponding parameter-method. The order and type is important. Do it
        //   like in the method 'filter1' above.

        //2. Pre-requisite: You have in your plugin project a class, which is derived
        //   from QMainWindow, QDialog or QWidget. This class can also include an ui-file,
        //   which has been designed using the QtDesigner. The class name is Widget1.

        // Create an instance of that class and return it.
        // The instance is deleted by the caller of the method 'widget1'.
        Widget1 *win = new Widget( /* your parameters */ );
        QWidget *widget = qobject_cast<QWidget*>(win); //cast it to QWidget, if it isn't already.
        if(widget == NULL)
        {
            retValue += ito::RetVal(ito::retError,0,tr("The widget could not be loaded").toLatin1().data());
        }
        return widget; //NULL in case of error
    }

The widget will then be shown if the user created it by the GUI or the widget is wrapped by an instance of the python class **ui**
(part of module **itom**). Then you can interact with elements of the widget using python like you can do it with every user interface
created with **QtDesigner**. You can also connect some python methods with signals of your widget or call slots of your widget. This is
also the same behaviour like dialogs have, which only have been created with **QtDesigner** and then loaded by an instance of class **ui**
using python.

Publish Filter- and Widget-Methods at Initialization
----------------------------------------------------

The most important step in the development of an algorithm plugin is to publish all created filter- and widget-methods. By that process,
the methods will be made available to |itom|, such that they can be used by the python scripting language, the GUI or other plugins. The publishing
is done in the method **init** of your plugin. It is important to say, that the two different signatures of a filter method (with or without observer)
need to be published in two different ways. A exemplary implementation is as follows:

.. code-block:: c++
    :linenos:

    ito::RetVal MyAlgoPlugin::init(QVector<ito::ParamBase> *paramsMand,
                                   QVector<ito::ParamBase> *paramsOpt,
                                   ItomSharedSemaphore *waitCond)
    {
        ItomSharedSemaphoreLocker locker(waitCond);

        ito::RetVal retval = ito::retOk;
        FilterDef *filter = NULL;
        AlgoWidgetDef *widget = NULL;

        //publish your filter-methods here, an example for a default filter definition (without status and cancellation):
        filter = new FilterDef(WLIfilter, WLIfilterParams,
                    tr("description").toLatin1().data(),  //description
                    ito::AddInAlgo::catNone,              //category
                    ito::AddInAlgo::iNotSpecified,        //interface
                    QString());                           //meta information for interface (e.g. file pattern to be loadable)
        m_filterList.insert("filterName", filter);

        //here an example for the 2nd filter definition (with status observer and cancellation):
        filter = new FilterDefExt(WLIfilterCancellable,
                    WLIfilterParams,
                    tr("description").toLatin1().data(), //description
                    ito::AddInAlgo::catNone,             //category
                    ito::AddInAlgo::iNotSpecified,       //interface
                    QString(),                           //meta information for interface (e.g. file pattern to be loadable)
                    true,                                //true if filter provides status information, else false
                    true);                               //true if filter listens to the interrupt flag
                                                       //of the observer and allows to be interrupted earlier, else false
        m_filterList.insert("filterName", filter);


        //publish your dialogs, main-windows, widgets... here, example:
        widget = new AlgoWidgetDef(widget1, widget1Params, tr("description").toLatin1().data(),
                                    ito::AddInAlgo::catNone, ito::AddInAlgo::iNotSpecified);
        m_algoWidgetList.insert("widgetName", widget);

        if (waitCond)
        {
            waitCond->returnValue = retval;
            waitCond->release();
        }

        return retval;
    }

For registering filter- and widget-methods, you have to create a new instance of the classes **FilterDef** /
**FilterDefExt** (with observer) or **AlgoWidgetDef** respectively and insert these newly created instances to the
maps **m_filterList** or **m_algoWidgetList** respectively. Their name in the map finally is the name of the filter
or widget and must be unique within |itom|; else the filter-method or widget-method can not be loaded at startup of |itom|.

The constructor of class **FilterDef** has the following arguments:

* *AddInAlgo::t_filter* **filterFunc** is the pointer to the static filter-method (*here:* filter1)
* *AddInAlgo::t_filterParam* **filterParamFunc** is the pointer to the corresponding parameter method (*here:* filter1Params)
* *QString* **description** is a description string of the filter
* *ito::AddInAlgo::tAlgoCategory* **category** is an optional value of the category enumeration, the filter belongs too (*default:* ito::AddInAlgo::catNone)
* *ito::AddInAlgo::tAlgoInterface* **interf** is an optional value of the interface enumeration, the filter fits to (*default:* ito::AddInAlgo::iNotSpecified)
* *QString* **interfaceMeta** is depending on the chosen interface an additional meta string (*default:* QString())

For a full reference of class **FilterDef**, see :ref:`plugin-Algo-FilterDef-Ref`.

The constructor of class **FilterDefExt** has the following arguments:

* *AddInAlgo::t_filterExt* **filterFuncExt** is the pointer to the static filter2-method (with observer) (*here:* filter2)
* *AddInAlgo::t_filterParam* **filterParamFunc** is the pointer to the corresponding parameter method (*here:* filter1Params)
* *QString* **description** is a description string of the filter
* *ito::AddInAlgo::tAlgoCategory* **category** is an optional value of the category enumeration, the filter belongs too (*default:* ito::AddInAlgo::catNone)
* *ito::AddInAlgo::tAlgoInterface* **interf** is an optional value of the interface enumeration, the filter fits to (*default:* ito::AddInAlgo::iNotSpecified)
* *QString* **interfaceMeta** is depending on the chosen interface an additional meta string (*default:* QString())
* *bool* **hasStatusInfo** indicates if the filter implements code to regularly report its progress to the passed observer
* *bool* **isCancellable** indicates if the filter implements code to regularly check the interrupt flag of the observer and stop the algorithm as soon as it has been set (e.g. via Python)

For a full reference of class **FilterDef**, see :ref:`plugin-Algo-FilterDefExt-Ref`.

The constructor of class **AlgoWidgetDef** has the following arguments:

* *AddInAlgo::t_algoWidget* **algoWidgetPFunc** is the pointer to the static widget-method (*here:* widget1)
* *AddInAlgo::t_filterParam* **algoWidgetParamFunc** is the pointer to the corresponding parameter method (*here:* widget1Params)
* *QString* **description** is a description string of the widget or GUI-element.
* *ito::AddInAlgo::tAlgoCategory* **category** is an optional value of the category enumeration, the widget-method belongs too (*default:* ito::AddInAlgo::catNone)
* *ito::AddInAlgo::tAlgoInterface* **interf** is an optional value of the interface enumeration, the widget-method fits to (*default:* ito::AddInAlgo::iNotSpecified)
* *QString* **interfaceMeta** is depending on the chosen interface an additional meta string (*default:* QString())

For a full reference of class **AlgoWidgetDef**, see :ref:`plugin-Algo-AlgoWidgetDef-Ref`.

For information about available algorithm categories and interfaces, see

.. toctree::

   plugin-algo-CatInterfaces.rst

The method-types **t_filter**, **t_algoWidget** and **t_filterParam** are defined by the following *typedef*:

.. code-block:: c++

    typedef ito::RetVal (* t_filter)(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut);
    typedef QWidget*    (* t_algoWidget)(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ito::RetVal &retValue);
    typedef ito::RetVal (* t_filterParam)(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut);


Finish and close plugin
-----------------------

Finally, the **close** method can be implemented, those structure is usually unchanged:

.. code-block:: c++
    :linenos:

    ito::RetVal MyAlgoPlugin::close(ItomSharedSemaphore *waitCond)
    {
        ItomSharedSemaphoreLocker locker(waitCond);
        ito::RetVal retval = ito::retOk;

        if (waitCond)
        {
            waitCond->returnValue = retval;
            waitCond->release();
        }

        return retval;
    }

The **close**-method does nothing but immediately releasing the parameer *waitCond*. For more information about
*ItomSharedSemaphore*, see :ref:`plugin-sharedSemaphore`.

.. toctree::
   :hidden:

   plugin-algo-Ref.rst
