/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef PCLFUNCTIONS_IMPL_H
#define PCLFUNCTIONS_IMPL_H

#include "../../common/sharedStructures.h"

#include "../../common/typeDefs.h"
#include "../../DataObject/dataobj.h"

namespace ito
{

class DataObject; //forward declaration

namespace pclHelper
{
    template<typename _Tp, int _Rows, int _Cols> ito::RetVal eigenMatrixToDataObj(const Eigen::Matrix<_Tp,_Rows,_Cols> &mat, DataObject &out)
    {
        ito::RetVal retval;
        ito::tDataType type;

        try
        {
            type = ito::getDataType2<_Tp*>();
        }
        catch(...)
        {
            retval += ito::RetVal(ito::retError, 0, "eigen matrix type cannot be converted to dataObject");
        }

        if(!retval.containsError())
        {
            const _Tp *data = mat.data();
            _Tp *rowPtr = NULL;
            size_t c = 0;
            size_t rows = mat.rows();
            size_t cols = mat.cols();
            out = ito::DataObject(rows, cols, type);

            if(mat.Options & Eigen::RowMajor)
            {
                for(size_t m = 0 ; m < rows ; m++)
                {
                    rowPtr = (_Tp*)out.rowPtr(0,m);
                    for(size_t n = 0 ; n < cols ; n++)
                    {
                        rowPtr[n] = data[c++];
                    }
                }
            }
            else
            {
                for(size_t m = 0 ; m < rows ; m++)
                {
                    rowPtr = (_Tp*)out.rowPtr(0,m);
                    for(size_t n = 0 ; n < cols ; n++)
                    {
                        rowPtr[n] = data[m + n * rows];
                    }
                }
            }

        }

        return retval;
    }


    template<typename _Tp, int _Rows, int _Cols> ito::RetVal dataObjToEigenMatrix(const DataObject &dataobj, Eigen::Matrix<_Tp,_Rows,_Cols> &mat)
    {
        ito::RetVal retval;
        ito::tDataType type;

        try
        {
            type = ito::getDataType2<_Tp*>();
        }
        catch(...)
        {
            retval += ito::RetVal(ito::retError, 0, "eigen matrix type is unknown for dataObject");
        }

        if(!retval.containsError())
        {
            ito::DataObject dobj;
            retval += dataobj.convertTo(dobj, type);
        }

        if (dataobj.getDims() != 2 || dataobj.getSize(0) != _Rows || dataobj.getSize(1) != _Cols)
        {
            retval += ito::RetVal(ito::retError, 0, "size of dataobj does not fit to requested Eigen::Matrix size");
        }

        if (!retval.containsError())
        {
            _Tp *data = mat.data();
            const _Tp *rowPtr = NULL;
            size_t c = 0;
            size_t rows = mat.rows();
            size_t cols = mat.cols();

            if(mat.Options & Eigen::RowMajor)
            {
                for(size_t m = 0 ; m < rows ; m++)
                {
                    rowPtr = (_Tp*)dataobj.rowPtr(0,m);
                    for(size_t n = 0 ; n < cols ; n++)
                    {
                        data[c++] = rowPtr[n];
                    }
                }
            }
            else
            {
                for(size_t m = 0 ; m < rows ; m++)
                {
                    rowPtr = (_Tp*)dataobj.rowPtr(0,m);
                    for(size_t n = 0 ; n < cols ; n++)
                    {
                        data[m + n * rows] = rowPtr[n];
                    }
                }
            }

        }

        return retval;
    }

} //end namespace pclHelper

} //end namespace ito

#endif
