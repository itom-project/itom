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

//#define ITOM_IMPORT_API
#include "../apiFunctionsInc.h"
//#undef ITOM_IMPORT_API
//#define ITOM_IMPORT_PLOTAPI
#include "../apiFunctionsGraphInc.h"
//#undef ITOM_IMPORT_PLOTAPI
#include "../abstractApiWidget.h"

#include <qevent.h>



namespace ito
{

//------------------------------------------------------------------------------------------------------------------------
class AbstractApiWidgetPrivate
{
public:
    AbstractApiWidgetPrivate() {}
};

//----------------------------------------------------------------------------------------------------------------------------------
AbstractApiWidget::AbstractApiWidget(QWidget *parent) :
    QWidget(parent),
    d(NULL)
{
    d = new AbstractApiWidgetPrivate();
}

//----------------------------------------------------------------------------------------------------------------------------------
AbstractApiWidget::~AbstractApiWidget()
{
    delete d;
    d = NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractApiWidget::setApiFunctionGraphBasePtr(void **apiFunctionGraphBasePtr)
{
    this->importItomApiGraph(apiFunctionGraphBasePtr);
    m_apiFunctionsGraphBasePtr = apiFunctionGraphBasePtr;
    ito::ITOM_API_FUNCS_GRAPH = apiFunctionGraphBasePtr;
}

//----------------------------------------------------------------------------------------------------------------------------------
void AbstractApiWidget::setApiFunctionBasePtr(void **apiFunctionBasePtr)
{
    this->importItomApi(apiFunctionBasePtr);
    m_apiFunctionsBasePtr = apiFunctionBasePtr;
    ito::ITOM_API_FUNCS = apiFunctionBasePtr;
}

//----------------------------------------------------------------------------------------------------------------------------------
bool AbstractApiWidget::event(QEvent *e)
{
    //the event User+123 is emitted by UiOrganizer, if the API has been prepared and can
    //transmitted to the plugin. This assignment cannot be done directly, since
    //the array ITOM_API_FUNCS is in another scope if called from itom. By sending an
    //event from itom to the plugin, this method is called and ITOM_API_FUNCS is in the
    //right scope. The methods above only set the pointers in the "wrong"-itom-scope (which
    //also is necessary if any methods of the plugin are directly called from itom).
    if (e->type() == (QEvent::User+123))
    {
        //importItomApi(m_apiFunctionsBasePtr);
        //importItomPlotApi(m_apiFunctionsGraphBasePtr);
        init();
    }

    return QWidget::event(e);
}


} //end namespace ito
