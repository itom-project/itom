/* ********************************************************************
itom measurement system
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2017, Institut fuer Technische Optik (ITO),
Universitaet Stuttgart, Germany

This file is part of itom.

itom is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

itom is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "paramEditorWidget.h"
#include <qevent.h>
#include <qheaderview.h>
#include <qaction.h>
#include <qlayout.h>

#include "common/addInInterface.h"

#include "qttreepropertybrowser.h"
#include "qtgroupboxpropertybrowser.h"
#include "qtpropertymanager.h"

class ParamEditorWidgetPrivate
{
	//Q_DECLARE_PUBLIC(ParamEditorWidget);

public:
    ParamEditorWidgetPrivate()
        : m_pBrowser(NULL),
        m_pIntManager(NULL)
    {};

    QPointer<ito::AddInBase> m_plugin;
    QtAbstractPropertyBrowser *m_pBrowser;
    QtIntPropertyManager *m_pIntManager;
};

//-----------------------------------------------------------------------
ParamEditorWidget::ParamEditorWidget(QWidget* parent /*= 0*/) : 
	QWidget(parent),
	d_ptr(new ParamEditorWidgetPrivate())
{
	Q_D(ParamEditorWidget);

	d->m_pBrowser = new QtTreePropertyBrowser();
    d->m_pIntManager = new QtIntPropertyManager(this);

    QHBoxLayout *hboxLayout = new QHBoxLayout();
    hboxLayout->addWidget(d->m_pBrowser);
    setLayout(hboxLayout);
}

//-----------------------------------------------------------------------
ParamEditorWidget::~ParamEditorWidget()
{
    Q_D(ParamEditorWidget);
    delete d->m_pBrowser;
}

//-----------------------------------------------------------------------
QPointer<ito::AddInBase> ParamEditorWidget::plugin() const
{
    Q_D(const ParamEditorWidget);
    return d->m_plugin;
}

//-----------------------------------------------------------------------
void ParamEditorWidget::setPlugin(QPointer<ito::AddInBase> plugin)
{
    Q_D(ParamEditorWidget);

    if (d->m_plugin.data() != plugin.data())
    {
        d->m_plugin = plugin;

        d->m_pBrowser->clear();

        if (plugin.isNull() == false)
        {
            QtGroupPropertyManager *groupManager = new QtGroupPropertyManager(this);
            d->m_pBrowser->addProperty(groupManager->addProperty("General"));
            d->m_pBrowser->addProperty(groupManager->addProperty("Test"));

            QMap<QString, ito::Param> *params;
            QMap<QString, ito::Param>::const_iterator iter;
            plugin->getParamList(&params);

            iter = params->constBegin();
            while (iter != params->constEnd())
            {
                if (iter->getType() == ito::ParamBase::Int)
                {
                    QtProperty *prop = d->m_pIntManager->addProperty(iter->getName());
                    prop->setEnabled(!(iter->getFlags() & ito::ParamBase::Readonly));
                    d->m_pIntManager->setValue(prop, iter->getVal<int>());
                    d->m_pIntManager->setMinimum(prop, iter->getMin());
                    d->m_pIntManager->setMaximum(prop, iter->getMax());
                    d->m_pBrowser->addProperty(prop);
                }
                ++iter;
            }
            
        }
    }
}