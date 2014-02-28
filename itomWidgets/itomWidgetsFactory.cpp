/* ********************************************************************
   itom measurement system
   URL: http://www.uni-stuttgart.de/ito
   Copyright (C) 2012, Institut für Technische Optik (ITO), 
   Universität Stuttgart, Germany 
 
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


#include <QtCore/QtPlugin>
#include "itomWidgetsFactory.h"

#include "factories/rangeSliderFactory.h"
#include "factories/collapsibleGroupBoxFactory.h"
#include "factories/doubleRangeSliderFactory.h"
#include "factories/doubleSliderFactory.h"
#include "factories/doubleSpinBoxFactory.h"
#include "factories/rangeSliderFactory.h"
#include "factories/rangeWidgetFactory.h"
#include "factories/sliderWidgetFactory.h"
#include "factories/pathLineEditFactory.h"
#include "factories/popupWidgetFactory.h"

//------------------------------------------------------------------------------------------------
ItomWidgetsFactory::ItomWidgetsFactory(QObject *parent)
    : QObject(parent)
{
    widgets.append(new CollapsibleGroupBoxFactory(this));
    widgets.append(new DoubleRangeSliderFactory(this));
    widgets.append(new DoubleSliderFactory(this));
    //widgets.append(new DoubleSpinBoxFactory(this));
    widgets.append(new RangeSliderFactory(this));
    widgets.append(new RangeWidgetFactory(this));
    widgets.append(new SliderWidgetFactory(this));
    widgets.append(new PathLineEditFactory(this));
    widgets.append(new PopupWidgetFactory(this));
}

//------------------------------------------------------------------------------------------------
QList<QDesignerCustomWidgetInterface*> ItomWidgetsFactory::customWidgets() const
{
    return widgets;
}

#if QT_VERSION <= 0x050000
    Q_EXPORT_PLUGIN2(itomWidgetsFactory, ItomWidgetsFactory)
#endif
