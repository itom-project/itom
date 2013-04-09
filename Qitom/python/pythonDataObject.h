/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#ifndef PYTHONDATAOBJECT
#define PYTHONDATAOBJECT

/* includes */
#include <string>
#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must before including global.h)
    #define NO_IMPORT_ARRAY

    //#define NPY_NO_DEPRECATED_API 0x00000007 //see comment in pythonNpDataObject.cpp
    //python
    // see http://vtk.org/gitweb?p=VTK.git;a=commitdiff;h=7f3f750596a105d48ea84ebfe1b1c4ca03e0bab3
    #ifdef _DEBUG
        #undef _DEBUG
        #if (defined linux) | (defined CMAKE)
            #include "Python.h"
            #include "numpy/arrayobject.h"
        #else
            #include "Python.h"
            #include "../Lib/site-packages/numpy/core/include/numpy/arrayobject.h" //for numpy arrays
        #endif
        #define _DEBUG
    #else
        #ifdef linux
            #include "Python.h"
            #include "numpy/arrayobject.h"
        #else
            #include "Python.h"
            #include "../Lib/site-packages/numpy/core/include/numpy/arrayobject.h" //for numpy arrays
        #endif
    #endif
#endif

#include "../../DataObject/dataobj.h"
#include <qobject.h>

namespace ito 
{
class PythonDataObject
    {

    public:

        //-------------------------------------------------------------------------------------------------
        // typedefs
        //------------------------------------------------------------------------------------------------- 
        typedef struct
        {
            PyObject_HEAD
            DataObject* dataObject;
            PyObject* base;
        }
        PyDataObject;

        

        typedef struct
        {
            char *name;
            int typeno;
        }
        PyDataObjectTypes;

        #define PyDataObject_Check(op) PyObject_TypeCheck(op, &PythonDataObject::PyDataObjectType)

        static inline void PyDataObject_SetBase( PyDataObject *op, PyObject *base )
        {
            PyObject* x = op->base; 
            Py_XINCREF(base); 
            op->base = base; 
            Py_XDECREF(x);
        }
        
        //-------------------------------------------------------------------------------------------------
        // constructor, deconstructor, alloc, dellaoc
        //------------------------------------------------------------------------------------------------- 

        static void PyDataObject_dealloc(PyDataObject *self);
        static PyObject *PyDataObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
        static int PyDataObject_init(PyDataObject *self, PyObject *args, PyObject *kwds);

        static int copyNpDataObjTags2DataObj(PyObject* npDataObject, DataObject* dataObj);
        static int parsePyObject2StdString(PyObject* pyObj, std::string &str);

        //-------------------------------------------------------------------------------------------------
        // general members
        //------------------------------------------------------------------------------------------------- 
        static PyObject *PyDataObject_name(PyDataObject *self);

        static PyObject* PyDataObject_repr(PyDataObject *self);

        static PyObject* PyDataObject_data(PyDataObject *self);
        static PyObject* PyDataObject_conj(PyDataObject *self);
        static PyObject* PyDataObject_conjugate(PyDataObject *self);
        static PyObject* PyDataObject_adj(PyDataObject *self);
        static PyObject* PyDataObject_adjugate(PyDataObject *self);
        static PyObject* PyDataObject_trans(PyDataObject *self);
        static PyObject* PyDataObject_copy(PyDataObject *self, PyObject* args);
        static PyObject* PyDataObject_astype(PyDataObject *self, PyObject* args);
        static PyObject* PyDataObject_size(PyDataObject *self, PyObject* args);
        static PyObject* PyDataObject_makeContinuous(PyDataObject *self);
        static PyObject* PyDataObject_locateROI(PyDataObject *self);
        static PyObject* PyDataObject_adjustROI(PyDataObject *self, PyObject* args);
        static PyObject* PyDataObject_squeeze(PyDataObject *self, PyObject* args);

        static PyObject* PyDataObject_mul(PyDataObject *self, PyObject *args);
        static PyObject* PyDataObject_div(PyDataObject *self, PyObject *args);

        static PyObject* PyDataObject_reshape(PyDataObject *self, PyObject *args);
    
        // Get / Set metadata / objecttags
        static PyObject* PyDataObj_SetAxisOffset(PyDataObject *self, PyObject *args);
        static PyObject* PyDataObj_SetAxisScale(PyDataObject *self, PyObject *args);
        static PyObject* PyDataObj_SetAxisDescription(PyDataObject *self, PyObject *args);
        static PyObject* PyDataObj_SetAxisUnit(PyDataObject *self, PyObject *args);

        static PyObject* PyDataObj_SetTag(PyDataObject *self, PyObject *args);
        static PyObject* PyDataObj_DeleteTag(PyDataObject *self, PyObject *args);
        static PyObject* PyDataObj_TagExists(PyDataObject *self, PyObject *args);
        static PyObject* PyDataObj_GetTagListSize(PyDataObject *self);
        static PyObject* PyDataObj_AddToProtocol(PyDataObject *self, PyObject *args);

        // end tags

        static PyObject* PyDataObject_RichCompare(PyDataObject *self, PyObject *other, int cmp_op);

        static PyGetSetDef PyDataObject_getseters[];
        static PyObject* PyDataObj_GetDims(PyDataObject *self, void *closure);
        static PyObject* PyDataObj_GetType(PyDataObject *self, void *closure);
        static PyObject* PyDataObj_GetContinuous(PyDataObject *self, void *closure);

        static PyObject* PyDataObject_getTagDict(PyDataObject *self, void *clousure);

        static PyObject* PyDataObject_getTags(PyDataObject *self, void *closure);
        static int PyDataObject_setTags(PyDataObject *self, PyObject *value, void *closure);
        static PyObject* PyDataObject_getAxisScales(PyDataObject *self, void *closure);
        static int PyDataObject_setAxisScales(PyDataObject *self, PyObject *value, void *closure);
        static PyObject* PyDataObject_getAxisOffsets(PyDataObject *self, void *closure);
        static int PyDataObject_setAxisOffsets(PyDataObject *self, PyObject *value, void *closure);
        static PyObject* PyDataObject_getAxisDescriptions(PyDataObject *self, void *closure);
        static int PyDataObject_setAxisDescriptions(PyDataObject *self, PyObject *value, void *closure);
        static PyObject* PyDataObject_getAxisUnits(PyDataObject *self, void *closure);
        static int PyDataObject_setAxisUnits(PyDataObject *self, PyObject *value, void *closure);
        static PyObject* PyDataObject_getValueUnit(PyDataObject *self, void *closure);
        static int PyDataObject_setValueUnit(PyDataObject *self, PyObject *value, void *closure);
        static PyObject* PyDataObject_getValueDescription(PyDataObject *self, void *closure);
        static int PyDataObject_setValueDescription(PyDataObject *self, PyObject *value, void *closure);
        static PyObject* PyDataObject_getValueScale(PyDataObject *self, void *closure);
        static PyObject* PyDataObject_getValueOffset(PyDataObject *self, void *closure);
        static PyObject* PyDataObject_getValue(PyDataObject *self, void *closure);
        static int PyDataObject_setXYRotationalMatrix(PyDataObject *self, PyObject *value, void *closure);        
        static PyObject* PyDataObject_getXYRotationalMatrix(PyDataObject *self, void *closure);

        static PyObject* PyDataObj_Array_StructGet(PyDataObject *self);
        static PyObject* PyDataObj_Array_Interface(PyDataObject *self);
        static PyObject* PyDataObj_Array_(PyDataObject *self, PyObject *args);

        static PyObject* PyDataObj_ToList(PyDataObject *self);
        static PyObject* PyDataObj_At(ito::DataObject *dataObj, unsigned int *idx);
        static PyObject* PyDataObj_At(ito::DataObject *dataObj, size_t continuousIdx);
        static PyObject* PyDataObj_ToListRecursive(ito::DataObject *dataObj, unsigned int *currentIdx, size_t iterationIndex);


        static void PyDataObj_Capsule_Destructor(PyObject* capsule); //called if capsule (dataObject exported by __array__struct_) is destroyed

        static PyObject* PyDataObj_Reduce(PyDataObject *self, PyObject *args);
        static PyObject* PyDataObj_SetState(PyDataObject *self, PyObject *args);

        //-------------------------------------------------------------------------------------------------
        // number protocol
        //------------------------------------------------------------------------------------------------- 
        static PyObject* PyDataObj_nbAdd(PyDataObject* o1, PyDataObject* o2);
        static PyObject* PyDataObj_nbSubtract(PyDataObject* o1, PyDataObject* o2);
        static PyObject* PyDataObj_nbMultiply(PyDataObject* o1, PyDataObject* o2);
        static PyObject* PyDataObj_nbRemainder(PyDataObject* o1, PyDataObject* o2);
        static PyObject* PyDataObj_nbDivmod(PyDataObject* o1, PyDataObject* o2);
        static PyObject* PyDataObj_nbPower(PyDataObject* o1, PyDataObject* o2, PyDataObject* o3);
        static PyObject* PyDataObj_nbNegative(PyDataObject* o1);
        static PyObject* PyDataObj_nbPositive(PyDataObject* o1);
        static PyObject* PyDataObj_nbAbsolute(PyDataObject* o1);
        static PyObject* PyDataObj_nbInvert(PyDataObject* o1);
        static PyObject* PyDataObj_nbLshift(PyDataObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbRshift(PyDataObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbAnd(PyDataObject* o1, PyDataObject* o2);
        static PyObject* PyDataObj_nbXor(PyDataObject* o1, PyDataObject* o2);
        static PyObject* PyDataObj_nbOr(PyDataObject* o1, PyDataObject* o2);

        static PyObject* PyDataObj_nbInplaceAdd(PyDataObject* o1, PyDataObject* o2);
        static PyObject* PyDataObj_nbInplaceSubtract(PyDataObject* o1, PyDataObject* o2);
        static PyObject* PyDataObj_nbInplaceMultiply(PyDataObject* o1, PyDataObject* o2);
        static PyObject* PyDataObj_nbInplaceRemainder(PyDataObject* o1, PyDataObject* o2);
        static PyObject* PyDataObj_nbInplacePower(PyDataObject* o1, PyDataObject* o2, PyDataObject* o3);
        static PyObject* PyDataObj_nbInplaceLshift(PyDataObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbInplaceRshift(PyDataObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbInplaceAnd(PyDataObject* o1, PyDataObject* o2);
        static PyObject* PyDataObj_nbInplaceXor(PyDataObject* o1, PyDataObject* o2);
        static PyObject* PyDataObj_nbInplaceOr(PyDataObject* o1, PyDataObject* o2);

        //-------------------------------------------------------------------------------------------------
        // iterator protocol
        //-------------------------------------------------------------------------------------------------
        static PyObject* PyDataObj_getiter(PyDataObject* self); //getiterfunc tp_iter;
        

        //-------------------------------------------------------------------------------------------------
        // mapping protocol
        //------------------------------------------------------------------------------------------------- 
        static int PyDataObj_mappingLength(PyDataObject* self);
        static PyObject* PyDataObj_mappingGetElem(PyDataObject* self, PyObject* key);
        static int PyDataObj_mappingSetElem(PyDataObject* self, PyObject* key, PyObject* value);

        //-------------------------------------------------------------------------------------------------
        // type structures
        //------------------------------------------------------------------------------------------------- 
        static PyMemberDef PyDataObject_members[];
        static PyMethodDef PyDataObject_methods[];
        static PyTypeObject PyDataObjectType;
        static PyModuleDef PyDataObjectModule;

        static PyNumberMethods PyDataObject_numberProtocol;
        static PyMappingMethods PyDataObject_mappingProtocol;

        //-------------------------------------------------------------------------------------------------
        // helper methods
        //-------------------------------------------------------------------------------------------------    
        static RetVal PyDataObj_ParseCreateArgs(PyObject *args, PyObject *kwds, int &typeno, std::vector<unsigned int> &dims, unsigned char &continuous);

        static PyDataObjectTypes PyDataObject_types[];
        static int typeNameToNumber(const char *name);
        static char* typeNumberToName(int typeno);

        static PyDataObject* createEmptyPyDataObject();

        static bool checkPyDataObject(int number, PyDataObject* o1 = NULL, PyDataObject* o2 = NULL, PyDataObject* o3 = NULL);

        static RetVal parseTypeNumber(int typeno, char &typekind, int &itemsize);
        static int parseTypeNumberInverse(char typekind, int itemsize);
        static int getTypenumOfCompatibleType(char typekind, int itemsize);

        //-------------------------------------------------------------------------------------------------
        // static type methods
        //-------------------------------------------------------------------------------------------------
        //static PyObject* PyDataObj_StaticArange(PyDataObject *self, PyObject *args);
        static PyObject* PyDataObj_StaticZeros(PyObject *self, PyObject *args, PyObject *kwds);
        static PyObject* PyDataObj_StaticOnes(PyObject *self, PyObject *args, PyObject *kwds);
        static PyObject* PyDataObj_StaticRand(PyObject *self, PyObject *args, PyObject *kwds);
        static PyObject* PyDataObj_StaticRandN(PyObject *self, PyObject *args, PyObject *kwds);
        static PyObject* PyDataObj_StaticEye(PyObject *self, PyObject *args /*, PyObject *kwds*/);


        //#################################################################################################
        // ITERATOR METHODS
        //#################################################################################################

        typedef struct
        {
            PyObject_HEAD
            size_t curIndex;
            size_t endIndex;
            PyObject* base;
        }
        PyDataObjectIter;

        //-------------------------------------------------------------------------------------------------
        // constructor, deconstructor, alloc, dellaoc
        //------------------------------------------------------------------------------------------------- 

        static void PyDataObjectIter_dealloc(PyDataObjectIter *self);
        static PyObject *PyDataObjectIter_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
        static int PyDataObjectIter_init(PyDataObjectIter *self, PyObject *args, PyObject *kwds);

        static PyObject* PyDataObjectIter_iternext(PyDataObjectIter* self);
        static PyObject* PyDataObjectIter_len(PyDataObjectIter* self);

        //-------------------------------------------------------------------------------------------------
        // type structures
        //------------------------------------------------------------------------------------------------- 
        static PyMethodDef PyDataObjectIter_methods[];
        static PyTypeObject PyDataObjectIterType;



};

} //end namespace ito

#endif
