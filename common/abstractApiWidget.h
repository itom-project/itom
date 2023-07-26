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

#ifndef ABSTRACTAPIWIDGET_H
#define ABSTRACTAPIWIDGET_H

#include "apiFunctionsGraphInc.h"
#include "apiFunctionsInc.h"

#include <qwidget.h>

#include "retVal.h"


#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

//place this macro in the header file of the designer plugin widget class right before the first section (e.g. public:)
#define WIDGET_ITOM_API \
    protected: \
        void importItomApi(void** apiPtr) \
        {ito::ITOM_API_FUNCS = apiPtr;} \
        void importItomApiGraph(void** apiPtr) \
        { ito::ITOM_API_FUNCS_GRAPH = apiPtr;} \
    public: \
        //.

class QEvent; //forward declaration

namespace ito {

class AbstractApiWidgetPrivate; //forward declaration

class ITOMCOMMONQT_EXPORT AbstractApiWidget : public QWidget
{
    Q_OBJECT

public:
    explicit AbstractApiWidget(QWidget *parent = 0);
    virtual ~AbstractApiWidget();

    virtual bool event(QEvent *e);
    void setApiFunctionGraphBasePtr(void **apiFunctionGraphBasePtr);
    void setApiFunctionBasePtr(void **apiFunctionBasePtr);
    void ** getApiFunctionGraphBasePtr(void) { return m_apiFunctionsGraphBasePtr; }
    void ** getApiFunctionBasePtr(void) { return m_apiFunctionsBasePtr; }

protected:
    virtual RetVal init() { return retOk; } //this method is called from after construction and after that the api pointers have been transmitted

    virtual void importItomApi(void** apiPtr) = 0;      /*!< function to provide access to the itom API functions. this methods are implemented in the plugin itsself. Therefore put the macro WIDGET_ITOM_API before the public section in the widget class. */
    virtual void importItomApiGraph(void** apiPtr) = 0; /*!< function to provide access to the itom API functions. this methods are implemented in the plugin itsself. Therefore put the macro WIDGET_ITOM_API before the public section in the widget class. */

    void **m_apiFunctionsGraphBasePtr;
    void **m_apiFunctionsBasePtr;

private:
    AbstractApiWidgetPrivate *d;

};

} // namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif // ABSTRACTAPIWIDGET_H
