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

#include "pythonAlgorithms.h"

#include "../../AddInManager/addInManager.h"
#include "../AppManagement.h"
#include "pythonEngine.h"
#include "pythonItom.h"
#include "pythonProgressObserver.h"
#include "pythonQtConversion.h"
#include "pythontParamConversion.h"

#include "opencv2/core/core_c.h"

namespace ito {

//-------------------------------------------------------------------------------------
void PythonAlgorithms::AddAlgorithmFunctions(PyObject* mod)
{
    if (PyType_Ready(&PythonAlgorithms::PyAlgorithmType) >= 0)
    {
        Py_INCREF(&PythonAlgorithms::PyAlgorithmType);
        if (PyModule_AddObject(mod, "algorithm", (PyObject*)&PythonAlgorithms::PyAlgorithmType) < 0)
        {
            Py_DECREF(&PythonAlgorithms::PyAlgorithmType);
        }
    }

    ito::AddInManager* aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    auto flist = aim->getFilterList();
    auto flistIt = flist->constBegin();
    QByteArray name;

    while (flistIt != flist->constEnd())
    {
        name = flistIt.key().toLatin1();

        PyObject *obj = PyUnicode_DecodeLatin1(name.data(), name.size(), nullptr);

        if (obj)
        {
            if (!PyUnicode_IsIdentifier(obj))
            {
                Py_DECREF(obj);
                qDebug() << "Algorithm " << name << " cannot be added to itom.algorithms module, since the name is no valid Python identifier.";
                continue;
            }

            Py_DECREF(obj);
        }

        PyObject* algoWrapper = createPyAlgorithm(flistIt.value());

        if (PyModule_AddObject(mod, name.data(), algoWrapper) < 0)
        {
            Py_DECREF(algoWrapper);
        }

        flistIt++;
    }
}

//-------------------------------------------------------------------------------------
PyObject* PythonAlgorithms::PyGenericAlgorithm(
    const QString& algorithmName, PyObject* self, PyObject* args, PyObject* kwds)
{
    ito::AddInManager* aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    auto flist = aim->getFilterList();
    auto cfit = flist->constFind(algorithmName);

    if (cfit == flist->constEnd())
    {
        PyErr_SetString(PyExc_ValueError, "Unknown filter, please check typing!");
        return nullptr;
    }

    ito::AddInAlgo::FilterDef* fFunc = cfit.value();

    return PyGenericAlgorithm2(fFunc, self, args, kwds);
}

//-------------------------------------------------------------------------------------
PyObject* PythonAlgorithms::PyGenericAlgorithm2(const AddInAlgo::FilterDef *filterDef, PyObject *self, PyObject *args, PyObject *kwds)
{
    ito::RetVal ret = ito::retOk;
    PythonEngine* pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    ito::AddInManager* aim = qobject_cast<ito::AddInManager*>(AppManagement::getAddInManager());
    const ito::AddInAlgo::FilterDefExt* fFuncExt = dynamic_cast<const ito::AddInAlgo::FilterDefExt*>(filterDef);

    QVector<ito::ParamBase> paramsMandBase, paramsOptBase, paramsOutBase;

    const ito::FilterParams* filterParams = aim->getHashedFilterParams(filterDef->m_paramFunc);

    if (filterParams == nullptr)
    {
        PyErr_SetString(PyExc_RuntimeError, "Parameters of filter could not be found.");
        return nullptr;
    }

    PyObject* kwdsArgs = nullptr;

    // check if pKwds contain the special argument name 'statusObserver' and if so obtain its value,
    // make a copy of pKwds without this argument and use this to parse the remaining parameters
    PyObject* statusObserverName = PyUnicode_FromString("_observer"); // new reference
    PyObject* statusObserver = kwds
        ? PyDict_GetItem(kwds, statusObserverName)
        : nullptr; // nullptr, if it does not contain, else: borrowed reference

    if (statusObserver)
    {
        kwdsArgs = PyDict_Copy(kwds); // new reference
        PyDict_DelItem(kwdsArgs, statusObserverName);
    }
    else
    {
        kwdsArgs = kwds;
        Py_XINCREF(kwdsArgs);
    }

    Py_XDECREF(statusObserverName);
    statusObserverName = nullptr;

    if (statusObserver)
    {
        if (!PyProgressObserver_Check(statusObserver))
        {
            Py_XDECREF(kwdsArgs);
            kwdsArgs = nullptr;
            PyErr_SetString(
                PyExc_RuntimeError,
                "Keyword-based parameter '_observer' must be of type itom.progressObserver");
            return nullptr;
        }
        else if (fFuncExt == nullptr)
        {
            if (PyErr_WarnEx(
                    PyExc_RuntimeWarning,
                    "Parameter '_observer' is given, but the called filter does not implement the "
                    "extended interface with additional status information",
                    1) == -1) // exception is raised instead of warning (depending on user defined
                              // warning levels)
            {
                Py_XDECREF(kwdsArgs);
                kwdsArgs = nullptr;
                return nullptr;
            }
        }
    }

    // parses python-parameters with respect to the default values given py (*it).paramsMand
    // and (*it).paramsOpt and returns default-initialized ParamBase-Vectors paramsMand and
    // paramsOpt.
    ret += parseInitParams(
        &(filterParams->paramsMand),
        &(filterParams->paramsOpt),
        args,
        kwdsArgs,
        paramsMandBase,
        paramsOptBase);

    // makes deep copy from default-output parameters (*it).paramsOut and returns it in paramsOut
    // (ParamBase-Vector)
    ret += copyParamVector(&(filterParams->paramsOut), paramsOutBase);

    Py_XDECREF(kwdsArgs);
    kwdsArgs = nullptr;

    if (ret.containsError())
    {
        PyErr_SetString(
            PyExc_RuntimeError, QObject::tr("Error while parsing parameters.").toLatin1().data());
        return nullptr;
    }

    // from here, python can do something else... (we assume that the filter
    // might be a longer operation)
    Py_BEGIN_ALLOW_THREADS

        QSharedPointer<ito::FunctionCancellationAndObserver>
            observer;

    if (fFuncExt)
    {
        if (statusObserver)
        {
            observer =
                *(((PythonProgressObserver::PyProgressObserver*)statusObserver)->progressObserver);
        }

        if (observer.isNull())
        {
            observer = QSharedPointer<ito::FunctionCancellationAndObserver>(
                new ito::FunctionCancellationAndObserver());
        }

        if (pyEngine)
        {
            pyEngine->addFunctionCancellationAndObserver(observer.toWeakRef());
        }
    }

    try
    {
        if (fFuncExt)
        {
            observer->reset();
            ret = (*(fFuncExt->m_filterFuncExt))(
                &paramsMandBase, &paramsOptBase, &paramsOutBase, observer);
        }
        else
        {
            ret = (*(filterDef->m_filterFunc))(&paramsMandBase, &paramsOptBase, &paramsOutBase);
        }
    }
    catch (cv::Exception& exc)
    {
        const char* errorStr = cvErrorStr(exc.code);

        ret += ito::RetVal::format(
            ito::retError,
            0,
            QObject::tr("OpenCV Error: %s (%s) in %s, file %s, line %d").toLatin1().data(),
            errorStr,
            exc.err.c_str(),
            exc.func.size() > 0 ? exc.func.c_str()
                                : QObject::tr("Unknown function").toLatin1().data(),
            exc.file.c_str(),
            exc.line);
        // see also cv::setBreakOnError(true) -> then cv::error(...) forces an access to 0x0000
        // (throws access error, the debugger stops and you can debug it)

        // use this to raise an access error that forces the IDE to break in this line (replaces
        // cv::setBreakOnError(true)).
#if defined _DEBUG
        static volatile int* p =
            0; // if your debugger stops in this line, another exception has been raised and you
               // have now the chance to see your callstack for debugging.
        *p = 0;
#endif
    }
    catch (std::exception& exc)
    {
        if (exc.what())
        {
            ret += ito::RetVal::format(
                ito::retError,
                0,
                QObject::tr("The exception '%s' has been thrown").toLatin1().data(),
                exc.what());
        }
        else
        {
            ret += ito::RetVal(
                ito::retError,
                0,
                QObject::tr("The exception '<unknown>' has been thrown").toLatin1().data());
        }
#if defined _DEBUG
        static volatile int* p =
            0; // if your debugger stops in this line, another exception has been raised and you
               // have now the chance to see your callstack for debugging.
        *p = 0;
#endif
    }
    catch (...)
    {
        ret += ito::RetVal(
            ito::retError,
            0,
            QObject::tr("An unspecified exception has been thrown").toLatin1().data());
#if defined _DEBUG
        static volatile int* p =
            0; // if your debugger stops in this line, another exception has been raised and you
               // have now the chance to see your callstack for debugging.
        *p = 0;
#endif
    }

    if (fFuncExt && pyEngine)
    {
        pyEngine->removeFunctionCancellationAndObserver(observer.data());
    }

    if (observer && observer->isCancelled() &&
        observer->cancellationReason() ==
            ito::FunctionCancellationAndObserver::ReasonKeyboardInterrupt)
    {
        ret = ito::retOk; // ignore the error message, since Python will raise a keyboardInterrupt
                          // though
    }

    Py_END_ALLOW_THREADS // now we want to get back the GIL from Python

    if (!PythonCommon::transformRetValToPyException(ret))
    {
        return nullptr;
    }
    else
    {
        if (paramsOutBase.size() == 0)
        {
            Py_RETURN_NONE;
        }
        else if (paramsOutBase.size() == 1)
        {
            return PythonParamConversion::ParamBaseToPyObject(paramsOutBase[0]); // new ref
        }
        else
        {
            // parse output vector to PyObject-Tuple
            PyObject* out = PyTuple_New(paramsOutBase.size());
            PyObject* temp;
            Py_ssize_t i = 0;
            bool error = false;

            foreach (const ito::ParamBase& p, paramsOutBase)
            {
                temp = PythonParamConversion::ParamBaseToPyObject(p); // new ref
                if (temp)
                {
                    PyTuple_SetItem(out, i, temp); // steals ref
                    i++;
                }
                else
                {
                    error = true;
                    break;
                }
            }

            if (error)
            {
                Py_DECREF(out);
                return nullptr;
            }
            else
            {
                return out;
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                              //
//          PYTHON MODULES - - - PYTHON TYPES - - - PYTHON MODULES //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PyMethodDef PythonAlgorithms::PythonMethodItomAlgorithms[] = {
    // "Python name", C Ffunction Code, Argument Flags, __doc__ description
    {nullptr, nullptr, 0, nullptr}};

PyModuleDef PythonAlgorithms::PythonModuleItomAlgorithms = {
    PyModuleDef_HEAD_INIT,
    "algorithms",
    nullptr,
    -1,
    PythonMethodItomAlgorithms,
    nullptr,
    nullptr,
    nullptr,
    nullptr};


//-------------------------------------------------------------------------------------
void PythonAlgorithms::PyAlgorithm_dealloc(PyAlgorithm* self)
{
    self->filterDef = nullptr;
    Py_TYPE(self)->tp_free((PyObject*)self);
};

//-------------------------------------------------------------------------------------
PyObject* PythonAlgorithms::PyAlgorithm_new(
    PyTypeObject* type, PyObject* /*args*/, PyObject* /*kwds*/)
{
    PyAlgorithm* self = (PyAlgorithm*)type->tp_alloc(type, 0);
    if (self != nullptr)
    {
        self->filterDef = nullptr;
    }

    return (PyObject*)self;
};

//-------------------------------------------------------------------------------------
int PythonAlgorithms::PyAlgorithm_init(PyAlgorithm* self, PyObject* args, PyObject* kwds)
{
    if (args == nullptr && kwds == nullptr)
    {
        return 0; // call from createPyAlgorithm
    }
    else
    {
        PyErr_Format(PyExc_ValueError, "_algorithm can never be directly called.");
        return -1;
    }
};

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonAlgorithms::PyAlgorithm_call(
    PyAlgorithm* self, PyObject* args, PyObject* kwds)
{
    return PyGenericAlgorithm2(self->filterDef, nullptr, args, kwds);
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonAlgorithms::PyAlgorithm_repr(PyAlgorithm* self)
{
    PyObject* result = PyUnicode_FromFormat(
        "algorithm(\"%s\", plugin \"%s\")",
        self->filterDef->m_name.toLatin1().data(),
        self->filterDef->m_pBasePlugin->objectName().toLatin1().data());
    return result;
}

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonAlgorithms::createPyAlgorithm(const ito::AddInAlgo::FilterDef* filterDef)
{
    PyAlgorithm* result =
        (PyAlgorithm*)PyObject_Call((PyObject*)&PyAlgorithmType, nullptr, nullptr);

    if (result != nullptr)
    {
        result->filterDef = filterDef;
        return (PyObject*)result; // result is always a new reference
    }
    else
    {
        Py_XDECREF(result);
        return nullptr;
    }
}

//-----------------------------------------------------------------------------
PyModuleDef PythonAlgorithms::PyAlgorithmModule = {
    PyModuleDef_HEAD_INIT,
    "_algorithm",
    "Algorithm wrapper.",
    -1,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr};

//-----------------------------------------------------------------------------
PyTypeObject PythonAlgorithms::PyAlgorithmType = {
    PyVarObject_HEAD_INIT(nullptr, 0) /* here has been nullptr,0 */
    "itom.algorithms._algorithm", /* tp_name */
    sizeof(PyAlgorithm), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)PyAlgorithm_dealloc, /* tp_dealloc */
    0, /* tp_print */
    0, /* tp_getattr */
    0, /* tp_setattr */
    0, /* tp_reserved */
    (reprfunc)PyAlgorithm_repr, /* tp_repr */
    0, /* tp_as_number */
    0, /* tp_as_sequence */
    0, /* tp_as_mapping */
    0, /* tp_hash  */
    (ternaryfunc)PyAlgorithm_call, /* tp_call */
    0, /* tp_str */
    0, /* tp_getattro */
    0, /* tp_setattro */
    0, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    0, /* tp_doc */
    0, /* tp_traverse */
    0, /* tp_clear */
    0, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    0, /* tp_iter */
    0, /* tp_iternext */
    0, /* tp_methods */
    0, /* tp_members */
    0, /* tp_getset */
    0,
    /* tp_base */ /*will be filled later before calling PyType_Ready */
    0, /* tp_dict */
    0, /* tp_descr_get */
    0, /* tp_descr_set */
    0, /* tp_dictoffset */
    (initproc)PyAlgorithm_init, /* tp_init */
    0,
    /* tp_alloc */ /*will be filled later before calling PyType_Ready */
    PyAlgorithm_new /* tp_new */
};

} // end namespace ito
