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

#ifndef PYTHONDATAOBJECT
#define PYTHONDATAOBJECT

/* includes */
#include <string>

#ifndef Q_MOC_RUN
    #define PY_ARRAY_UNIQUE_SYMBOL itom_ARRAY_API //see numpy help ::array api :: Miscellaneous :: Importing the api (this line must before including global.h)
    #define NO_IMPORT_ARRAY

    #include "pythonWrapper.h"
#endif

#include "../../DataObject/dataobj.h"
//#include <qobject.h>

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
            const char *name;
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
        static PyObject* PyDataObject_copy(PyDataObject *self, PyObject* args, PyObject *kwds);
        static PyObject* PyDataObject_astype(PyDataObject *self, PyObject* args, PyObject* kwds);
        static PyObject* PyDataObject_normalize(PyDataObject *self, PyObject* args, PyObject* kwds);
        static PyObject* PyDataObject_size(PyDataObject *self, PyObject* args);
        static PyObject* PyDataObject_makeContinuous(PyDataObject *self);
        static PyObject* PyDataObject_locateROI(PyDataObject *self);
        static PyObject* PyDataObject_adjustROI(PyDataObject *self, PyObject* args, PyObject *kwds);
        static PyObject* PyDataObject_squeeze(PyDataObject *self, PyObject* args);

        static PyObject* PyDataObject_mul(PyDataObject *self, PyObject *args);
        static PyObject* PyDataObject_div(PyDataObject *self, PyObject *args);

        static PyObject* PyDataObject_reshape(PyDataObject *self, PyObject *args, PyObject *kwds);

        static PyObject* PyDataObject_createMask(PyDataObject *self, PyObject *args, PyObject* kwds);

        // Get / Set metadata / objecttags
        static PyObject* PyDataObj_SetAxisOffset(PyDataObject *self, PyObject *args);
        static PyObject* PyDataObj_SetAxisScale(PyDataObject *self, PyObject *args);
        static PyObject* PyDataObj_SetAxisDescription(PyDataObject *self, PyObject *args);
        static PyObject* PyDataObj_SetAxisUnit(PyDataObject *self, PyObject *args);
        static PyObject* PyDataObj_PhysToPix(PyDataObject *self, PyObject *args, PyObject *kwds);
        static PyObject* PyDataObj_PixToPhys(PyDataObject *self, PyObject *args, PyObject *kwds);
		static PyObject* PyDataObj_CopyMetaInfo(PyDataObject *self, PyObject *args, PyObject *kwds);

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
        static PyObject* PyDataObj_GetShape(PyDataObject *self, void *closure);
        static PyObject* PyDataObj_GetContinuous(PyDataObject *self, void *closure);

        static PyObject* PyDataObject_getTagDict(PyDataObject *self, void *closure);
        static int PyDataObject_setTagDict(PyDataObject *self, PyObject *value, void *closure);

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

		static PyObject* PyDataObject_getReal(PyDataObject *self, void *closure);
		static int PyDataObject_setReal(PyDataObject *self, PyObject *value, void *closure);

		static PyObject* PyDataObject_getImag(PyDataObject *self, void *closure);
		static int PyDataObject_setImag(PyDataObject *self, PyObject *value, void *closure);

		static PyObject* PyDataObject_abs(PyDataObject *self, void *closure);
		static PyObject* PyDataObject_arg(PyDataObject *self, void *closure);

        static int PyDataObject_setXYRotationalMatrix(PyDataObject *self, PyObject *value, void *closure);
        static PyObject* PyDataObject_getXYRotationalMatrix(PyDataObject *self, void *closure);

        static PyObject* PyDataObject_getValue(PyDataObject *self, void *closure);
        static int PyDataObject_setValue(PyDataObject *self, PyObject *value, void *closure);

        static PyObject* PyDataObject_transpose(PyDataObject *self, void *closure);

        static PyObject* PyDataObj_Array_StructGet(PyDataObject *self);
        static PyObject* PyDataObj_Array_Interface(PyDataObject *self);
        static PyObject* PyDataObj_Array_(PyDataObject *self, PyObject *args);

        static PyObject* PyDataObject_real(PyDataObject *self);
        static PyObject* PyDataObject_imag(PyDataObject *self);

        static PyObject* PyDataObj_ToGray(PyDataObject *self, PyObject *args, PyObject *kwds);
        static PyObject* PyDataObj_ToNumpyColor(PyDataObject *self, PyObject *args, PyObject *kwds);
        static PyObject* PyDataObj_SplitColor(PyDataObject *self, PyObject *args, PyObject *kwds);

        static PyObject* PyDataObj_ToList(PyDataObject *self);
        static PyObject* PyDataObj_At(ito::DataObject *dataObj, const unsigned int *idx);
        static PyObject* PyDataObj_ToListRecursive(ito::DataObject *dataObj, unsigned int *currentIdx, int iterationIndex);


        static void PyDataObj_Capsule_Destructor(PyObject* capsule); //called if capsule (dataObject exported by __array__struct_) is destroyed

        static PyObject* PyDataObj_Reduce(PyDataObject *self, PyObject *args);
        static PyObject* PyDataObj_SetState(PyDataObject *self, PyObject *args);
		static PyObject* PyDataObj_lineCut(PyDataObject *self, PyObject *args);

        //-------------------------------------------------------------------------------------------------
        // number protocol
        //
        // python note: Binary and ternary functions must check the type of all their operands, and implement
        //    the necessary conversions (at least one of the operands is an instance of the defined type).
        //    If the operation is not defined for the given operands, binary and ternary functions must return
        //    Py_NotImplemented, if another error occurred they must return NULL and set an exception.
        //-------------------------------------------------------------------------------------------------
        static PyObject* PyDataObj_nbAdd(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbSubtract(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbMultiply(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbMatrixMultiply(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbDivide(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbRemainder(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbDivmod(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbPower(PyObject* o1, PyObject* o2, PyObject* o3);
        static PyObject* PyDataObj_nbNegative(PyObject* o1);
        static PyObject* PyDataObj_nbPositive(PyObject* o1);
        static PyObject* PyDataObj_nbAbsolute(PyObject* o1);
        static PyObject* PyDataObj_nbInvert(PyObject* o1);
        static PyObject* PyDataObj_nbLshift(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbRshift(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbAnd(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbXor(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbOr(PyObject* o1, PyObject* o2);
        static int PyDataObj_nbBool(PyDataObject *self);
        static PyObject* PyDataObj_nbInplaceAdd(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbInplaceSubtract(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbInplaceMultiply(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbInplaceRemainder(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbInplacePower(PyObject* o1, PyObject* o2, PyObject* o3);
        static PyObject* PyDataObj_nbInplaceLshift(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbInplaceRshift(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbInplaceAnd(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbInplaceXor(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbInplaceOr(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbInplaceTrueDivide(PyObject* o1, PyObject* o2);
        static PyObject* PyDataObj_nbInplaceMatrixMultiply(PyObject* o1, PyObject* o2);

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
        static int dObjTypeFromName(const char *name);
        static const char* typeNumberToName(int typeno);
        static int numDataTypes();

        static PyDataObject* createEmptyPyDataObject();
        static PyObject* createPyDataObjectFromArray(PyObject *npArray); //returns NULL with set Python exception if npArray could not be converted to data object

        static bool checkPyDataObject(int number, PyObject* o1 = NULL, PyObject* o2 = NULL, PyObject* o3 = NULL);

        static RetVal parseTypeNumber(int typeno, char &typekind, int &itemsize);
        static int getDObjTypeOfNpArray(char typekind, int itemsize);
        static int getCompatibleDObjTypeOfNpArray(char typekind, int itemsize);
        static int getNpTypeFromDataObjectType(int type);
        static std::string getNpDTypeStringFromNpDTypeEnum(const int type);

        static ito::RetVal copyNpArrayValuesToDataObject(PyArrayObject *npNdArray, ito::DataObject *dataObject, ito::tDataType type);
        static int PyDataObj_CreateFromShapeTypeData(PyDataObject *self, PyObject *args, PyObject *kwds); //helper method for PyDataObject_init
        static int PyDataObj_CreateFromNpNdArrayAndType(PyDataObject *self, PyObject *args, PyObject *kwds, bool addNpOrgTags); //helper method for PyDataObject_init
        static bool PyDataObj_CopyFromDatetimeNpNdArray(PyDataObject *self, PyArrayObject *dateTimeArray, int dims, const int* sizes);
        static bool PyDataObj_CopyFromTimedeltaNpNdArray(PyDataObject *self, PyArrayObject *timeDeltaArray, int dims, const int* sizes);


        //-------------------------------------------------------------------------------------------------
        // static type methods
        //-------------------------------------------------------------------------------------------------
        //static PyObject* PyDataObj_StaticArange(PyDataObject *self, PyObject *args);
        static PyObject* PyDataObj_StaticZeros(PyObject *self, PyObject *args, PyObject *kwds);
        static PyObject* PyDataObj_StaticOnes(PyObject *self, PyObject *args, PyObject *kwds);
		static PyObject* PyDataObj_StaticNans(PyObject *self, PyObject *args, PyObject *kwds);
        static PyObject* PyDataObj_StaticRand(PyObject *self, PyObject *args, PyObject *kwds);
        static PyObject* PyDataObj_StaticRandN(PyObject *self, PyObject *args, PyObject *kwds);
        static PyObject* PyDataObj_StaticEye(PyObject *self, PyObject *args, PyObject *kwds);
        static PyObject* PyDataObj_StaticFromNumpyColor(PyObject *self, PyObject *args, PyObject *kwds);
		static PyObject* PyDataObj_dstack(PyObject *self, PyObject *args, PyObject *kwds);



        //#################################################################################################
        // ITERATOR METHODS
        //#################################################################################################

        typedef struct
        {
            PyObject_HEAD
            ito::DObjConstIterator it;
            ito::DObjConstIterator itEnd;
            Py_ssize_t len;
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
