/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2019, Institut fuer Technische Optik (ITO),
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

#include "pythonProgressObserver.h"

#include "../global.h"
#include <qsharedpointer.h>
#include "pythonUi.h"
#include "../organizer/uiOrganizer.h"
#include "../AppManagement.h"


//------------------------------------------------------------------------------------------------------

namespace ito
{

//------------------------------------------------------------------------------------------------------
void PythonProgressObserver::PyProgressObserver_addTpDict(PyObject * tp_dict)
{
}

//------------------------------------------------------------------------------------------------------
void PythonProgressObserver::PyProgressObserver_dealloc(PyProgressObserver* self)
{
    DELETE_AND_SET_NULL(self->progressObserver);
    Py_TYPE(self)->tp_free((PyObject*)self);
};

//------------------------------------------------------------------------------------------------------
PyObject* PythonProgressObserver::PyProgressObserver_new(PyTypeObject *type, PyObject* /*args*/, PyObject* /*kwds*/)
{
    PyProgressObserver* self = (PyProgressObserver*)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->progressObserver = NULL;
    }

    return (PyObject *)self;
};

//------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------
PyDoc_STRVAR(PyProgressObserver_doc,"progressObserver(progressBar : uiItem = None, label : uiItem = None, progressMinimum : int = 0, progressMaximum : int = 100) -> creates a progressObserver object. \n\
\n\
This class is a wrapper for the class QFont of Qt. It provides possibilities for creating a font type. \n\
\n\
Parameters \n\
----------- \n\
family : {str} \n\
    The family name may optionally also include a foundry name, e.g. 'Helvetica [Cronyx]'. If the family is available \
    from more than one foundry and the foundry isn't specified, an arbitrary foundry is chosen. If the family isn't \
    available a family will be set using a best-matching algorithm. \n\
pointSize : {int}, optional \n\
    If pointSize is zero or negative, the point size of the font is set to a system-dependent default value. \
    Generally, this is 12 points, except on Symbian where it is 7 points. \n\
weight : {int}, optional \n\
    Weighting scale from 0 to 99, e.g. font.Light, font.Normal (default), font.DemiBold, font.Bold, font.Black \n\
italic : {bool}, optional \n\
    defines if font is italic or not (default)");
int PythonProgressObserver::PyProgressObserver_init(PyProgressObserver *self, PyObject *args, PyObject * kwds)
{
    PyObject *progressBar = NULL;
    PyObject *label = NULL;
    int progressMinimum = 0;
    int progressMaximum = 100;

    const char *kwlist[] = {"progressBar", "label", "progressMinimum", "progressMaximum", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "|O!O!ii", const_cast<char**>(kwlist), &PythonUi::PyUiItemType, &progressBar, &PythonUi::PyUiItemType, &label, &progressMinimum, &progressMaximum))
    {
        return -1;
    }

    self->progressObserver = new QSharedPointer<ito::FunctionCancellationAndObserver>(new ito::FunctionCancellationAndObserver(progressMinimum, progressMaximum));

    PythonUi::PyUiItem *p = (PythonUi::PyUiItem*)progressBar;
    PythonUi::PyUiItem *l = (PythonUi::PyUiItem*)label;

    UiOrganizer *uiOrg = (UiOrganizer*)AppManagement::getUiOrganizer();

    QSharedPointer<unsigned int> objectID(new unsigned int);

    ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());


    QMetaObject::invokeMethod(uiOrg, "connectWidgetsToProgressObserver", Q_ARG(bool, p != NULL), Q_ARG(unsigned int, p ? p->objectID : 0), 
        Q_ARG(bool, l != NULL), Q_ARG(unsigned int, l ? l->objectID : 0), 
        Q_ARG(QSharedPointer<ito::FunctionCancellationAndObserver>, *(self->progressObserver)), 
        Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

    if (!locker.getSemaphore()->wait(PLUGINWAIT))
    {
        PyErr_SetString(PyExc_RuntimeError, "Timeout while connecting widgets to progressObserver");
        return -1;
    }

    if (!PythonCommon::transformRetValToPyException(locker.getSemaphore()->returnValue))
    {
        return -1;
    }

    return 0;
};

//-----------------------------------------------------------------------------
/*static*/ PyObject* PythonProgressObserver::PyProgressObserver_repr(PyProgressObserver *self)
{
    PyObject *result = PyUnicode_FromFormat("progressObserver()");
    
    return result;
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(progressObserver_getProgressMinimum_doc, "Gets the minimum value of the progress. \n\
\n\
The minimum progress value is the minimum scalar value that the observed function or algorithm should set as its lowest progress value.");
PyObject* PythonProgressObserver::PyProgressObserver_getProgressMinimum(PyProgressObserver *self, void * /*closure*/)
{
    if (!self || self->progressObserver == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "progressObserver is not available");
        return NULL;
    }

    return PyLong_FromLong((*(self->progressObserver))->progressMinimum());
}

//-----------------------------------------------------------------------------
PyDoc_STRVAR(progressObserver_getProgressMaximum_doc, "Gets the maximum value of the progress. \n\
\n\
The maximum progress value is the maximum scalar value that the observed function or algorithm should set as its highest progress value.");
PyObject* PythonProgressObserver::PyProgressObserver_getProgressMaximum(PyProgressObserver *self, void * /*closure*/)
{
    if (!self || self->progressObserver == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "progressObserver is not available");
        return NULL;
    }

    return PyLong_FromLong((*(self->progressObserver))->progressMaximum());
}


//-----------------------------------------------------------------------------
PyGetSetDef PythonProgressObserver::PyProgressObserver_getseters[] = {
    {"progressMinimum", (getter)PyProgressObserver_getProgressMinimum,       (setter)NULL, progressObserver_getProgressMinimum_doc, NULL},
    {"progressMaximum", (getter)PyProgressObserver_getProgressMaximum,       (setter)NULL, progressObserver_getProgressMaximum_doc, NULL},
    {NULL}  /* Sentinel */
};

//-----------------------------------------------------------------------------
PyMethodDef PythonProgressObserver::PyProgressObserver_methods[] = {
    {NULL}  /* Sentinel */
};




//-----------------------------------------------------------------------------
PyModuleDef PythonProgressObserver::PyProgressObserverModule = {
    PyModuleDef_HEAD_INIT, "progressObserver", "Registers a label and / or progress bar to visualize the progress of a function call within algorithm plugins", -1,
    NULL, NULL, NULL, NULL, NULL
};

//-----------------------------------------------------------------------------
PyTypeObject PythonProgressObserver::PyProgressObserverType = {
    PyVarObject_HEAD_INIT(NULL,0) /* here has been NULL,0 */
    "itom.progressObserver",             /* tp_name */
    sizeof(PyProgressObserver),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)PyProgressObserver_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    (reprfunc)PyProgressObserver_repr,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    PyProgressObserver_doc,              /* tp_doc */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
    0,                           /* tp_richcompare */
    0,                           /* tp_weaklistoffset */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    PyProgressObserver_methods,          /* tp_methods */
    0,                         /* tp_members */
    PyProgressObserver_getseters,        /* tp_getset */
    0,                         /* tp_base */ /*will be filled later before calling PyType_Ready */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyProgressObserver_init,                       /* tp_init */
    0,                         /* tp_alloc */ /*will be filled later before calling PyType_Ready */
    PyProgressObserver_new     /* tp_new */
};



} //end namespace ito