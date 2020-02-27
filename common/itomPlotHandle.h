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

#ifndef ITOMPLOTHANDLE_H
#define ITOMPLOTHANDLE_H

#ifdef __APPLE__
extern "C++" {
#endif

    /* includes */
#include "commonGlobal.h"
#include <string>

    namespace ito
    {

        //----------------------------------------------------------------------------------------------------------------------------------
        /** @class ItomPlotHandle
        *   @brief  class for a interval type containing the plot handle / unique id and name.
        *
        *   This class is used to allow a conversion between Py_uiItem and C++/Qt.
        */
        class ItomPlotHandle
        {
        public:
            ItomPlotHandle()
            {
                m_objectID = 0;
                m_pObjName = "";
                m_pWidgetClassName = "";
            }
            explicit ItomPlotHandle(const char* objName, const char* widgetClassName, const unsigned int objectID)
            {
                m_objectID = objectID;
                if (objName)
                {
                    m_pObjName = std::string(objName);
                }
                else
                {
                    m_pObjName = "";
                }

                if (widgetClassName)
                {
                    m_pWidgetClassName = std::string(widgetClassName);
                }
                else
                {
                    m_pWidgetClassName = "";
                }

            }
            ItomPlotHandle(const ItomPlotHandle &rhs)
            {
                m_objectID = rhs.m_objectID;
                m_pObjName = rhs.m_pObjName;
                m_pWidgetClassName = rhs.m_pWidgetClassName;
            }

            ~ItomPlotHandle()
            {
                m_pObjName = "";
                m_pWidgetClassName = "";
                m_objectID = 0;
            }

            inline std::string getObjName() const { return m_pObjName; }
            inline std::string getWidgetClassName() const { return m_pWidgetClassName; }
            inline unsigned int getObjectID() const { return m_objectID; }
        private:
            std::string m_pObjName;
            std::string m_pWidgetClassName;
            unsigned int m_objectID;
        };

    } //end namespace ito

#ifdef __APPLE__
}
#endif

#endif
