/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom.

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef PYTHONITOMMETAOBJECT_H
#define PYTHONITOMMETAOBJECT_H

#include "../global.h"

#include <qbytearray.h>
#include <qmetaobject.h>

namespace ito
{

    struct PythonQObjectMarshal
    {
        PythonQObjectMarshal() : m_objectID(0), m_object(NULL) {}

        PythonQObjectMarshal(QByteArray objName, const char* className, QObject *object) :
            m_objName(objName),
            m_objectID(0),
            m_object(object)
        {
            m_className = QByteArray(className);
        }

        PythonQObjectMarshal(QObject *obj) :
            m_objName(obj->objectName().toLatin1()),
            m_objectID(0),
            m_object(obj)
        {
            m_className = QByteArray(obj->metaObject()->className());
        }

        QByteArray m_objName;
        QByteArray m_className;
        unsigned int m_objectID;
        QObject *m_object; //casted from QObject
    };



// MethodDescription
class MethodDescription
{
public:
    MethodDescription();
    MethodDescription(
        QByteArray &name,
        QByteArray &signature,
        QMetaMethod::MethodType type,
        QMetaMethod::Access access,
        int methodIndex,
        int retType,
        int nrOfArgs,
        int *argTypes
    );
    MethodDescription(QMetaMethod &method);
    MethodDescription(const MethodDescription &copy);
    ~MethodDescription();

    MethodDescription &operator=(const MethodDescription &other);

    inline bool isValid() const { return (m_methodIndex >= 0); }    /*!< returns true if member m_methodIndex is 0 or bigger, hence, method is valid */
    inline QMetaMethod::MethodType type() const { return m_type; }  /*!< returns method-type (\a m_type) */
    inline QMetaMethod::Access access() const { return m_access; }  /*!< returns access-value (\a m_access) */
    inline int methodIndex() const { return m_methodIndex; }        /*!< returns method index */
    inline QByteArray name() const { return m_name; }               /*!< returns method's name */
    inline int retType() const { return m_retType; }                /*!< returns return value type */
    inline int nrOfArgs() const { return m_nrOfArgs; }              /*!< returns number of arguments */
    inline int* argTypes() const { return m_argTypes; }             /*!< returns allocated integer array with type-id of every argument */
    inline QByteArray signature() const { return m_signature; }     /*!< returns full normalized signature of this method */

    inline bool checkMethod(QByteArray &name, int nrOfArgs) const { return (name == m_name && nrOfArgs == m_nrOfArgs); }

private:
    QByteArray m_name;               /*!< name of method */
    int m_methodIndex;               /*!< index of signal, slot or method */
    QByteArray m_signature;          /*!< complete normalized signature of this method, slot or signal */
    QMetaMethod::MethodType m_type;  /*!< 0:method, 1:signal, 2:slot, 3:constructor (see QMetaMethod::MethodType) */
    QMetaMethod::Access m_access;    /*!< 0:private, 1:protected, 2:public (see QMetaMethod::Access) */
    int m_retType;                   /*!< type-id (see QMetaType) of return value (0 if void) */
    int m_nrOfArgs;                  /*!< number of arguments this method, signal or slot has */
    int *m_argTypes;                 /*!< integer-array with size m_nrOfArgs containing the type-id of every argument (see QMetaType) */

};

typedef QList<MethodDescription> MethodDescriptionList;

/*!
    \class FctCallParamContainer
    \brief each instance of this class contains the parameters (including return parameter) for any function call,
        which is parsed by the Qt-signal-slot system. This class is especially used for wrapping function calls between
        C++ and any python method. The convention for the main member variables corresponds to the usual Qt-way to wrap such
        function calls.

        Each parameter is stored in this container using the construct and destroy method of QMetaType. The result of deeply copying
        a variable by QMetaType::construct is a void* containing the deep-copy and the corresponding type-id (integer). Both are then
        saved in a void* and int-array in the corresponding order. The first value of these arrays is always reserved for the return value.
        If the return value is of type void, the type-id of the first element is equal to zero.
*/
class FctCallParamContainer
{
public:
    //! constructor
    /*!
        initializes the FctCallParamContainer with a given number of arguments. Each argument is not allocated, hence
        each value in \ref m_argTypes is set to -1 and each value in \ref m_args is set to NULL.

        \param nrOfParams is the number of arguments (without return value) for the corresponding method / function
    */
    FctCallParamContainer(int nrOfParams) :
      m_nrOfParams(nrOfParams),
      m_sizeArgs(nrOfParams+1)
    {
        m_args = new void*[m_sizeArgs];
        m_argTypes = new int[m_sizeArgs];

        for (int i = 0; i < m_sizeArgs; i++)
        {
            m_args[i] = nullptr;
            m_argTypes[i] = -1;
        }
    };

    //! destructor
    /*!
        Each value in \ref m_args is destroyed using QMetaType::destroy, since this FctCallParamContainer always holds a deep copy of
        each parameter.
    */
    ~FctCallParamContainer()
    {
        for (int i = 0; i < m_sizeArgs; i++)
        {
            QMetaType(m_argTypes[i]).destroy(m_args[i]);
        }

        DELETE_AND_SET_NULL_ARRAY(m_argTypes);
        DELETE_AND_SET_NULL_ARRAY(m_args);
    }

    inline void** args() { return m_args; };                 /*!< returns \ref m_args */
    inline int* argTypes() { return m_argTypes; };           /*!< returns \ref m_argTypes */
    inline int getRetType() const { return m_argTypes[0]; }; /*!< returns type of return value */

    //! initializes the return value
    /*!
        At first, an existing return value is deleted and then the new return value given by type
        is created. The value is always the default type of this argument.

        \param type is the desired type-id of the default value which is assumed as return type
    */
    inline void initRetArg(int type)  //reference of ptr is stolen and will be deleted by this class
    {
        if (m_args[0])
        {
            QMetaType(m_argTypes[0]).destroy(m_args[0]);
        }

        m_argTypes[0] = type;
        m_args[0] = QMetaType(type).create(nullptr);
    };

    //! stores a pair of variable-type and corresponding void-pointer as parameter with given index number
    /*!
        At first any existing parameter at the given index position is deleted. Then the new value is saved.
        Consider, that no further deep copy of the new variable is created, hence, the reference to the void-pointer
        is stolen and will be deleted by this FctCallParamContainer. You can create an appropriate pair of void-pointer
        and type-id using QMetaType::construct.

        \param index is the parameter index (zero-based)
        \param ptr is the void-pointer with the internal data of the parameter
        \param type is the corresponding type-id with respect to QMetaType
    */
    inline void setParamArg(unsigned int index, void* ptr, int type)  //reference of ptr is stolen and will be deleted by this class
    {
        if ((int)index < 0 || (int)index >= m_nrOfParams)
        {
            return;
        }

        if (m_args[index + 1])
        {
            QMetaType(m_argTypes[index + 1]).destroy(m_args[index + 1]);
        }

        m_args[index+1] = ptr;
        m_argTypes[index+1] = type;
    };

private:
    //! copy constructor is not accessible
    FctCallParamContainer( const FctCallParamContainer & /*copy*/ ) = delete;
    int m_nrOfParams;  /*!<  number of arguments (hence \ref m_sizeArgs - 1) */
    int m_sizeArgs;  /*!< number of arguments + 1 (for return value), hence length of \ref m_args and \ref m_argTypes */
    void** m_args;   /*!< void*-array containing the data of each parameter, NULL if the parameter is not available / filled (yet) */
    int* m_argTypes; /*!< int-array with the corresponding type-id's for each parameter in \ref m_args */
};

} //end namespace ito

#endif // PYTHONITOMMETAOBJECT_H
