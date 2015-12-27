.. include:: ../../include/global.inc

.. sectionauthor:: Marc Gronle

.. _plugin-Params:

Parameter-Container class of |itom|
===================================

Introduction
------------

The base idea behind a container-class for parameters of varying types is to pass these parameters to 
methods without the need of extensive templating. Of course, both |Python| and |Qt| provide such classes, 
namely **PyObject** and **QVariant**. Nevertheless, |itom| provides an own, low-level container
class, which is not dependent on any 3rd party library. This container only provides support for some specific
types, that are widely used within |itom|. This parameter container is mainly used for the whole 
communication process between all types of plugins and |itom|. Examples for their use are:

* Parameters of each plugin, which can be read by *getParam* or set by *setParam*. Additionally these parameters of the plugin can be obtained by the python command :py:func:`getParamList()`.
* Mandatory and optional parameters for the constructor of plugin instances.
* Parameter transfer between configuration dialog, docking widget toolbox and plugin itself.

The whole implementation of the container-class is found in the files *sharedStructures.h* and 
*sharedStructures.cpp*, both lying in the folder *common*, where you can also find the file 
*addInInterface.h*.

The possible types, covered by this container are:

* **char** (Type 1)
* **integer** (Type 1)
* **double** (Type 1)
* **char-array** (Type 1)
* **integer-array** (Type 1)
* **double-array** (Type 1)
* **string** (zero-terminated, Type 1)
* **DataObject** (pointer-only, Type 2)
* **PointCloud** (pointer-only, Type 2)
* **PolygonMesh** (pointer-only, Type 2)
* **HWRef** (pointer-only, points to an instance of AddInBase, Type 2)

.. note::

    *Type 1* means, that these types are internally copied when calling a constructor, copy constructor
    or assignement operator of the parameter, where for parameters of *Type 2* only the pointer to the value
    itself is internally stored in the parameter, hence, only this pointer is copied at the above mentioned
    methods. The reason is, that a quick passing of the parameter is provided, on the other side, parameters
    of *Type 2* need some further attention concerning thread-safety and/or creation and deletion
    responsability.

*ParamBase* and *Param* and the *Meta*-classes
----------------------------------------------

There are different classes, defined in *sharedStructures.h* which can be used for the parameter
container:

* class **ParamBase** represents a pure multi-type container, which only contains the name of the parameter and its value. This is the quickest parameter container implementation and should always be used, if only the parameter should be passed and no further information are needed. In any other cases use an implementation of the derived class *Param*.

* class **Param** is derived from *ParamBase* and additionally contains a description string for the value (optional) and some further meta-information (optional) in form of an internal pointer, that points to an instance of class *MetaParam* or one of its derived classes.

* class **MetaParam** is the base class for all parameter type-dependent meta information classes, like *IntMeta*, *DoubleMeta*, *StringMeta*... The idea is to add some restrictions about value ranges, allowed values... to parameters if this is needed or available. Please consider, that the meta information class is not internally checked in the parameter classes, hence, you can assign every value you want to. However the programmer, that is using a parameter, can access the MetaParam-instance and program its own validator for the parameter or use one of the pre-defined methods.

* class **CharMeta**, **IntMeta**, **DoubleMeta** are meta information classes derived from MetaParam which contain a minimum and maximum value for the parameter. Parameters of array-types may also contain an instance of one of these classes in order to describe the allowed range of every element of the array.

* class **StringMeta** provides further information for the parameter of type *String* such that you restrict the string to certain values, which also can be evaluated in the sens of a regular expression or wildcard expression.

* class **HWMeta** provides restrictive information for a parameter of type *HWRef*

For more information about the meta information classes see :ref:`plugin-ParamsMeta`.

Usage and differences of the classes *ParamBase* and *Param*
------------------------------------------------------------

Variables of class **Param** are used whenever you explicitly want to add further information to your
parameter. Examples might be:

* Vectors of mandatory and optional parameters, used as template for creating an instance of a plugin.
* Vectors of mandatory and optional parameters, used as template for creating a widget defined in a plugin of type *AddInAlgo*.
* Vectors of mandatory, optional and out parameters, used as template for calling a filter.
* Plugin-internal parameters, stored in the Map *m_params*.
* *getParam*-method of plugins, which usually return one specific value of map *m_params*. Here an instance of class *Param* is returned and not *ParamBase* such that advanced information about the value can be presented.
* Vector of out-parameters of filters.

Variables of class *ParamBase* are used when you only need to transfer the parameter itself:

* Method *setParam* of plugins. The validation of the given value is done with respect to its corresponding value in map *m_params*.
* Vector of mandatory and optional parameters used for calling the constructor (method *init*) of plugins.
* Vector of mandatory, optional and out parameters used for calling filters in plugins.
* Vector of mandatory and optional parameters used for calling widgets, defined in plugins of type *AddInAlgo*.

In the case of the described mandatory and optional parameter vectors, |itom| is requesting the template version (class *Param*)
form the plugin and has enough information, in order to check the user input (done in GUI or by python) with respect to the template.
Finally a vector of type *ParamBase* is created, where all the default values, given by the templates, are overwritten by the user input.
Then the filter, plugin constructor or widget constructor is called with the version of *ParamBase*.


.. _plugin-params-typesFlags:

Types and flags
---------------

The type as well as additional flags of each parameter is defined by an OR combination of values, contained
in the enumeration **ito::ParamBase::Type**. The last 16 bit (bit 1-16) of this enumeration are reserved for
type information, the first 16 bits may contain flags.

They can be separated using an AND-operation with the masks **ito::paramFlagMask** or **ito::paramTypeMask**:

.. code-block:: c++
    
    int typesAndFlags
    int types = typesAndFlags & ito::paramTypeMask
    int flags = typesAndFlags & ito::paramFlagMask

The following (high-level) types are available:

.. code-block:: c++
    
    enum Type {
        ...
        //type (bit 1-16)
        Pointer         = 0x000001, //do not use directly
        Char            = 0x000002, //Character-Parameter
        Int             = 0x000004, //Integer-Parameter
        Double          = 0x000008, //Double-Parameter
        String          = 0x000020 | Pointer, //zero-terminated String-Parameter
        HWRef           = 0x000040 | Pointer | NoAutosave, //pointer to plugin-instance
        DObjPtr         = 0x000010 | Pointer | NoAutosave, //pointer to dataObject
        CharArray       = Char     | Pointer, //array of characters
        IntArray        = Int      | Pointer, //array of integers
        DoubleArray     = Double   | Pointer, //array of doubles
        PointCloudPtr   = 0x000080 | Pointer | NoAutosave, //pointer to point cloud
        PointPtr        = 0x000100 | Pointer | NoAutosave, //pointer to point
        PolygonMeshPtr  = 0x000200 | Pointer | NoAutosave  //pointer to polygon mesh
        ...
    };

.. note::
    
    All pointer-based types have the **NoAutosave**-flag, since a pointer can not be saved to harddrive. The
    arrays of **CharArray**, **IntArray** and **DoubleArray** are internally copied (e.g. in a copy-constructor),
    therefore only use them for smaller arrays and not for matrices with millions of entries. This might be an inefficient
    structure though.

The following flags are implemented in the **Type**-enumeration:

.. code-block:: c++
    
    enum Type {
        NoAutosave,
        Readonly,
        In,
        Out
        ...
    };

The behaviour of the **NoAutosave**-flag can be read in see :ref:`plugin-autoloadsave-policy`.
The **readonly**-flag marks this parameter to be readonly. Please consider, that this flag is not evaluated in the classes
**Param** or **ParamBase**, but the programmer has access to this flag must implement the necessary behaviour. The flags **In**
and **Out** or their combination are important for the declaration of the default parameters for plugins or filter-calls. If none
of them is set, the flag **In** is automatically set. **In** indicates, that the parameter is handled like an input-variable only, hence,
the filter or plugin's init method will not change the value of this parameter. A variable of type **In|Out** passes a value and the value
might be changed within a filter call. This is a suitable form to pass a dataObject whose content and size might be changed by the filter.
Parameters with flag **Out** only are only accepted in the parameter vector which is the default for the output-variables of a filter...
It is only allowed to mark parameter of type **Char**, **Int**, **Double**, **String**, **CharArray**, **IntArray** or **DoubleArray** as **Out**-
parameters.
    

Class *ParamBase*
-----------------

The class :cpp:class:`ParamBase` consists of the following main elements or member variables, which however are only accessible by
corresponding getter- or setter-methods:

.. c:member:: m_type
    
    This variable contains an OR combination of the data type, covered by the parameter container as well as some
    additional flags (read-only, auto-save). Read the section :ref:`plugin-params-typesFlags` for more information about the type.
    
    The type part of this member is obtained by **getType()**, the flags can be obtained by **getFlags()**.

.. c:member:: ito::ByteArray m_name
    
     This member contains the name of the parameter. This name is for example used for accessing the parameter in the python's *setParam* or *getParam* method and usually you can also use this name as keyword in a python argument list of appropriate method calls.
     
     Access the name of a parameter by using **getName()**. This returns the zero-terminated name string as char-pointer.
     
.. c:member:: values
    
    There are three further member variables which are used in order to store the variable content of the parameter container.
    Reading and writing these values is only done by the constructor or the methods **getVal<_Tp>** and **setVal<_Tp>**.

Typical creations for parameters of class **ParamBase** are:

.. code-block:: c++
    
    //empty parameter (name: NULL, type: 0)
    ParamBase p1;
    
    //creating an integer-parameter, flag: In, value: 2
    ParamBase p2("IntParam", ito::ParamBase::Int | ito::ParamBase::In, 2);
    
    //creating a double-parameter, flag: Readonly, value: -4.0
    ParamBase p3("Name", ito::ParamBase::Double | ito::ParamBase::Readonly, -4.0);
    
    //creating a string-parameter
    ParamBase p4("Name", ito::ParamBase::String, "default text");
    
    //creating an integer-array parameter
    int size = 5;
    int* a = new int[size];
    //.. fill a with valid values
    ParamBase p5("Array", ito::ParamBase::IntArray, size, a);
    //a is copied, therefore delete it now
    delete[] a;
    a = NULL;
    
    //passing a dataObject pointer as parameter
    ito::DataObject *dObj = new ito::DataObject(...);
    ParamBase p6("param", ito::ParamBase::DObjPtr, dObj);
    //be careful: p6 only holds a pointer to dObj, therefore you can only delete it
    // if p6 does not access it any more.
    
    //passing a pointer to another actuator- or dataIO-instance to a parameter
    ito::AddInActuator *aia = ...;
    ParamBase p7("motor", ito::ParamBase::HWRef, aia);
    //like with the dataObject. Be careful and make sure, that the pointer 'aia' remains
    //accessible during the lifetime of p7.

The parameter **p1** has no suitable type or value right now. However, you can assign another parameter to **p1** 
by using the assignment operator:

.. code-block:: c++
    
    p1 = ParamBase("newVal", ito::ParamBase::Char, 128)

If you have an array-parameter, you can access one single index of this array, which is then returned as new
instance of **ParamBase**. If the index is out of range or the parameter is no array-type, an empty instance
of **ParamBase** is returned:

.. code-block:: c++
    
    //use p5 from the example above
    ParamBase p5_0 = p5[0] //is a valid parameter of type Int
    ParamBase p5_5 = p5[5] //error. empty ParamBase since index exceeded the maximum size.

Reading values from the parameter is done by the method **getVal**. This method must be called with a template
parameter, that corresponds to the original data type, which is covered by the parameter.

.. code-block:: c++
    
    //This example is based on the constructed params above
    int p2_val = p2.getVal<int>();
    
    double p3_val = p3.getVal<double>();
    
    //the following examples return the internal pointer to the string or arrays.
    //This pointer is no copy, therefore you are not allowed to delete the pointer.
    char* p4_val = p4.getVal<char*>();
    
    int* p5_val = p5.getVal<int*>();
    //you can access the elements of p5 by
    int temp = p5_val[0];
    temp = p5_val[4];
    //p5_val[5] is not allowed, since it exceeds the number of elements of this array
    
    //in order to get the number of values in the parameter, use the following 
    //implementation
    int length = 0;
    p5_val = p5.getVal<int*>(length);
    //now length is equal to 5.
    
    //if you don't want to change the content of the pointer based parameter but only read the content,
    //consider to get the value as const parameter:
    const int* p5_val_const = p5.getVal<const int*>();
    
    //pointer-parameters are obtained by using the right template
    //parameter of the getVal method. The internal pointer of the
    //parameter is then casted to the template type.
    ito::DataObject *dObj = p6.getVal<ito::DataObject*>();
    const ito::DataObject *dObjConst = p6.getVal<ito::DataObject*>();
    
    ito::AddInActuator *aia = p7.getVal<ito::AddInActuator*>();
    
    //If you are sure that the parameter contains at least any plugin, however you have no idea
    //whether it is an acutator or an instance of dataIO, you could at first get the
    //base instance to ito::AddInBase and then try to safely cast it to your requested type:
    ito::AddInBase *aib = p7.getVal<ito::AddInBase*>();
    ito::AddInActutator *aia = qobject_cast<ito::AddInActuator*>(aib);
    //aia is NULL, if the cast failed.

If the given template parameter does not fit to the corresponding parameter type, the value of the
parameter will be casted to the given template type. If this is not possible an exception is raised.
The exception is of type **std::logic_error**.

Settings values to the parameter can be analogously done by the method **setVal**. This also is a template
based method. The following code snippets show examples how to change values of the previously constructed
parameters **p1** to **p7**:

.. code-block:: c++
    
    ito::RetVal retValue;
    retValue += p2.setVal<int>(5); //retValue remains retOk
    retValue += p3.setVal<double>(-3.7); //retValue remains retOk
    
    //p4 is a string-type. New values assigned to p4 (here: "new value")
    // are internally copied.
    retValue += p4.setVal<char*>("new value"); //retValue remains retOk
    
    //for array-types, you can only assign the whole array and not change any
    //elements. For changing values, use getVal<_Tp>(...) in order to obtain the pointer and
    //change the values directly.
    int values[] = {1,2,3,4,5};
    retValue += p5.setVal<int*>(values,5); //always provide the length of the array
    //again, the setVal-method above internally copies the array and you can destroy the 
    //source.
    int length = p5.getLen(); //length of array
    
    ito::DataObject dObj;
    retValue += p6.setVal<ito::DataObject*>( &dObj );
    //again remember, that p6 requires, that dObj remains accessible during the lifetime of p6.
    
.. note::
    
    The method **setVal** will never change the assigned type of the parameter. If the new value
    can not be converted into the internal type of the parameter, **setVal** will return with a **RetVal**,
    that contains errors.

If you only want to copy the content of one parameter of type **ParamBase** to your parameter, then you
can use the method **copyValueFrom**, which requires another instance of **ParamBase** or **Param** (since this is derived
from **ParamBase**). The method returns an error if the parameters are not compatible:

.. code-block::c++
    
    ParamBase p1("other parameter",ParamBase::Int,5);
    ParamBase p2("your parameter", ParamBase::Int,4);
    RetVal retValue = p2.copyValueFrom(&p1);
    //retValue is retOk
    int value = p2.getVal<int>(); //value is 5 now
    
    ParamBase p3("wrong", ParamBase::String);
    retValue += p3.copyValueFrom(&p1);
    //retValue.containsError() returns true
    
For a full reference to all member function of class **ParamBase**, see :ref:`plugin-paramBase-Ref`.

Class *Param*
-------------

The class **Param** is derived from :cpp:class`ParamBase`. Therefore it has all features of **ParamBase** including two additional
member variables:

    .. cpp:member:: ParamMeta *m_pMeta
        
        This is a pointer to a struct containing type-dependent meta information about this plugin. This pointer may also be
        NULL, if no meta information is provided. The meta-information struct is always owned by the parameter and deeply copied
        when calling for instance a copy constructor. For more information see :ref:`plugin-ParamsMeta`.
        
        Access to the meta information struct is given by
        
        .. code-block:: c++
            
            Param p;
            ParamMeta* meta = p.getMeta();
            //in this case meta is NULL, since no meta information has been set to 'p'.
            
            //now we create a integer-variable with a min and max value
            Param p2("var1",ParamBase::Int,2);
            IntMeta meta2(-2,2);
            
            //now we set the meta information of p2 to meta2. Since the ownership of meta2 should not
            //be taken by p2, the second argument is false. Then, p2 makes a copy of meta2.
            p2.setMeta(&meta2,false);
            
            meta = p.getMeta();
            //meta is now a pointer to a structure of type ParamMeta. It can be casted to IntMeta.

    .. cpp:member:: ito::ByteArray m_Info
        
        This is the description string of the parameter. If no description is indicated, this pointer is NULL, else it is a
        zero-terminated string, which is also copied, when the parameter is called using a copy constructor or assigned to another
        parameter.
        
        The description can be obtained by
        
        .. code-block:: c++
            
            Param p("name", ParamBase::String, "content", "information")
            const char* descr = p.getInfo(); //descr is 'information'
        
        .. note::
            
            Do not delete the char-pointer returned by **getInfo**, since this is only a reference to the internal description
            string of the parameter.
        
        The description string is changed by
        
        .. code-block:: c++
            
            p.setInfo("new information")

The full reference of class :cpp:class:`Param` is available in :ref:`plugin-param-Ref`.

In the following, examples about how to create parameters and meta information of different types are shown:

* **Integer-Type (Type: Int)**
    
    This is one fixed-point number in the integer-range.
    
    .. code-block:: c++
        
        //integer value between 0 and 10, default: 5
        ito::Param param("intNumber", ito::ParamBase::Int, 0, 10, 5, "description");
        //this default constructor automatically creates an internal meta-information struct
        //of class IntMeta.
        
        // or (here param becomes owner of IntMeta-instance)
        ito::Param param("intNumber", ParamBase::Int, 5, new IntMeta(0,10), "description");
        
        // or (integer-variable without meta information)
        ito::Param param("intNumber", ParamBase::Int, 5, NULL, "description");
        param.setMeta(new IntMeta(0,10), true); //take ownership of IntMeta-instance
        
        int value = param.getVal<int>();                //returns 5
        ito::RetVal retValue = param.setVal<int>(6);    //returns ito::retOk
        bool numeric = param.isNumeric()                //returns true, since integer is a numeric value.
        
        // accessing the min-max-value is obtained by getting the IntMeta-struct
        IntMeta *meta = dynamic_cast<IntMeta*>(param.getMeta());
        if(meta) //meta is only valid, if it has been assigned.
        {
            int min = meta->getMin()     //returns 0
            int max = meta->getMax()     //returns 10
        }
        int len = param.getLen()                        //returns 1
    
* **Double-Type (Type: Double)**
    
    This is one floating-point number in the double-range.
    
    .. code-block:: c++
        
        //integer value between 0.0 and 10.0, default: 5.0
        ito::Param param("doubleNumber", ito::ParamBase::Double, 0.0, 10.0, 5.0, "description");
        //this default constructor automatically creates an internal meta-information struct
        //of class DoubleMeta.
        
        // or (here param becomes owner of DoubleMeta-instance)
        ito::Param param("doubleNumber", ParamBase::Double, 5.0, DoubleMeta::all(), "description");
        // the command DoubleMeta::all() creates a new instance of DoubleMeta, where the boundaries
        // are the minimum and maximum possible value of the double-range.
        
        // or (double-variable without meta information)
        ito::Param param("doubleNumber", ParamBase::Double, 5.0, NULL, "description");
        param.setMeta(new DoubleMeta(0.0,10.0), true); //take ownership of DoubleMeta-instance
        
        double value = param.getVal<double>();               //returns 5.0
        ito::RetVal retValue = param.setVal<double>(6.0);    //returns ito::retOk
        bool numeric = param.isNumeric()                     //returns true, since integer is a numeric value.
        
        // accessing the min-max-value is obtained by getting the DoubleMeta-struct
        DoubleMeta *meta = dynamic_cast<DoubleMeta*>(param.getMeta());
        if(meta) //meta is only valid, if it has been assigned.
        {
            double min = meta->getMin()
            double max = meta->getMax()
        }
        int len = param.getLen()                        //returns 1
        
* **String-Type (Type: String)**
    
    This is one zero-terminated String.
    
    .. code-block:: c++
        
        ito::Param param("string", ito::ParamBase::String, "", "description");
        
        //if you want to provide a string-meta information, you must do it in the following separate lines:
        ito::StringMeta *meta = new ito::StringMeta(ito::StringMeta::String);
        meta->addItem("yes");
        meta->addItem("no");
        //the meta information indicates, that only the exact matches of both values "yes" or "no"
        //might be accepted by this parameter.
        param.setMeta(meta, true); //takes ownership of meta
        
        char* value = param.getVal<char*>();    //returns the pointer to the internally saved string.
        ito::RetVal retValue = param.setVal<char*>("test"); //should return ito::retOk, String is copied to internal storage.
        bool numeric = param.isNumeric()        //returns false
        int len = param.getLen()                //0 if empty string, else length of string
        
    .. note::
        
        Please do not delete the pointer to the internally saved string, obtained by getVal<char*>()!
        
* **Array of char values (Type: CharArray)**
    
    This is an array of character values. Consider that you should use the constructor where you can give the length of the array, else an error is returned.
    
    .. code-block:: c++
        
        char ptr = [0,56,127,-10,-20];
        ito::Param param("array", ito::ParamBase::CharArray, 5, &ptr, "description");
        //you can add a meta-information struct of class CharMeta to that char-array (if desired)
        
        char* value = param.getVal<char*>();    //returns the pointer to the first element of the array
        ito::RetVal retValue = param.setVal<char*>(ptr,5); //should return ito::retOk
        bool numeric = param.isNumeric()        //returns false (even it is an array of numeric values)
        int len = param.getLen()                //5
        ito::Param param0 = param[0];           //returns a char-parameter with value 0
        
* **Array of integer values (Type: IntArray)**
    
    This is an array of integer values. Consider that you should use the constructor where you can give the length of the array, else an error is returned.
    
    .. code-block:: c++
        
        int ptr = [1,2,3,4,5];
        ito::Param param("array", ito::ParamBase::IntArray, 5, &ptr, "description");
        //you can add a meta-information struct of class IntMeta to that integer array (if desired)
        
        int* value = param.getVal<int*>();    //returns the pointer to the first element of the array
        ito::RetVal retValue = param.setVal<int*>(ptr,5); //should return ito::retOk
        bool numeric = param.isNumeric()        //returns false (even it is an array of numeric values)
        int len = param.getLen()                //5
        ito::ParamBase param2 = param[1]        //returns integer-parameter (casted to ParamBase) of second item, value: 2
        
* **Array of integer values (Type: DoubleArray)**
    
    This is an array of double values. Consider that you should use the constructor where you can give the length of the array, else an error is returned.
    
    .. code-block:: c++
        
        double ptr = [1.2,2.3,3.4,4.1,5.2];
        ito::Param param("array", ito::ParamBase::DoubleArray, 5, &ptr, "description");
        
        double* value = param.getVal<double*>();    //returns the pointer to the first element of the array
        ito::RetVal retValue = param.setVal<double*>(ptr,5); //should return ito::retOk
        bool numeric = param.isNumeric()        //returns false (even it is an array of numeric values)
        int len = param.getLen()                //5

* **Reference to any initialized instance of dataIO or actuator (Type: HWRef)**
    
    This is the reference to any other initialized instance of dataIO or actuator. The called method should check whether the instance has the necessary properties or type. Consider, that the flag *NoAutosave* is always set for that type. If such a parameter is passed to the **init**-method of a plugin, the reference of the passed plugin is automatically increased
    (marks the plugin as being used by the new plugin) and vice-versa decremented when the new plugin is closed again.
    
    You can restrict the allowed plugin-references to types which have a minimum amount of bits in the plugin's type bitmask set. Further you can decline a restriction by indicating
    the exact name of a plugin. Please consider, that this check is passed in form of a class **HWMeta**, however the check is not executed by classes **Param** or **ParamBase**.
    You have to check this manually.
    
    .. code-block:: c++
        
        ito::Param param("serialPort", ito::ParamBase::HWRef, NULL, "description");
        
        //additionally define the meta-information
        ito::HWMeta *meta = new ito::HWMeta("SerialIO"); //restriction to plugins with name "SerialIO"
        param.setMeta(meta, true); //takes ownership of meta-pointer. Do not delete meta from that point on.
        
        //returns the hardware pointer casted to ito::AddInBase*
        ito::AddInBase* value = param.getVal<ito::AddInBase*>();   
        
        ito::RetVal retValue = param.setVal<void*>(ptr); //should return ito::retOk
        bool numeric = param.isNumeric()        //returns false
        int len = param.getLen()                //-1, since no length available
        
* **Reference to any initialized instance of DataObject (Type: DObjPtr)**
    
    This is the reference to an instance of *DataObject*. The called method should check whether the instance has the necessary properties or type. Consider, that the flag *NoAutosave* is always set for that type. You can further give a meta information struct of class *DObjMeta* in order to specify the data object more in detail.
    
    .. code-block:: c++
        
        ito::DataObject *dObj = new ito::DataObject();
        ito::Param param("image", ito::ParamBase::DObjPtr, dObj, "description");
        
        //create a meta information where you only allow 2-dim data objects of type (u)int8.
        //The necessary check is not automatically executed. You have to manually program it.
        ito::DObjMeta *meta = new ito::DObjMeta(ito::tUInt8 | ito::tInt8, 2, 2);
        param.setMeta(meta,true);
        
        //returns the pointer casted to DataObject*
        ito::DataObject* value = param.getVal<ito::DataObject*>();    
        ito::RetVal retValue = param.setVal<void*>(ptr); //should return ito::retOk
        bool numeric = param.isNumeric()        //returns false
        int len = param.getLen()                //-1, since no length available
        
        //if you marked the parameter to be an in-parameter only (flag ito::ParamBase::In set and ito::ParamBase::Out is not set)
        //please get the DataObject only in a const version:
        const ito::DataObject *dObjConst = param.getVal<const ito::DataObject*>();
        
        //if you do not need param again, you can delete dObj:
        delete dObj;
        dObj = NULL;
        
* **Reference to any initialized instance of ito::pclPointCloud (Type: PointCloudPtr)**
    
    This is the reference to an instance of pclPointCloud. The called method should check whether the instance has the necessary properties or type. Consider, that the flag *typeNoAutosave* is always set for that type.
    
    .. code-block:: c++
        
        ito::Param param("pcl", ito::ParamBase::PointCloudPtr, NULL, "description");
        
        //returns the pointer casted to pclPointCloud*
        ito::pclPointCloud* value = param.getVal<ito::pclPointCloud*>();    
        ito::RetVal retValue = param.setVal<ito::pclPointCloud*>(ptr); //should return ito::retOk
        bool numeric = param.isNumeric()        //returns false
        int len = param.getLen()                //-1, since no length available
        
* **Reference to any initialized instance of ito::pclPoint (Type: PointPtr)**
    
    This is the reference to an instance of *pclPoint*. The called method should check whether the instance has the necessary properties or type. Consider, that the flag *NoAutosave* is always set for that type. Currently, there is no meta information struct available for that type.
    
    .. code-block:: c++
        
        ito::Param param("point", ito::ParamBase::PointPtr, NULL, "description");
        
        //returns the pointer casted to pclPoint*
        ito::pclPoint* value = param.getVal<ito::pclPoint*>();    
        ito::RetVal retValue = param.setVal<ito::pclPoint*>(ptr); //should return ito::retOk
        bool numeric = param.isNumeric()        //returns false
        int len = param.getLen()                //-1, since no length available
        
* **Reference to any initialized instance of ito::pclPolygonMesh (Type: PolygonMeshPtr)**
    
    This is the reference to an instance of *pclPolygonMesh*. The called method should check whether the instance has the necessary properties or type. Consider, that the flag *NoAutosave* is always set for that type. Currently, there is no meta information struct available for that type.
    
    .. code-block:: c++
        
        ito::Param param("polygonMesh", ito::ParamBase::PolygonMeshPtr, NULL, "description");
        
        //returns the pointer casted to pclPolygonMesh*
        ito::pclPolygonMesh* value = param.getVal<ito::pclPolygonMesh*>();    
        ito::RetVal retValue = param.setVal<ito::pclPolygonMesh*>(ptr); //should return ito::retOk
        bool numeric = param.isNumeric()        //returns false
        int len = param.getLen()                //-1, since no length available
        
        
.. toctree::
   :hidden:

   plugin-paramBase-Ref.rst
   plugin-param-Ref.rst