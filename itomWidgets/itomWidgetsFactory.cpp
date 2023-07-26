/* ********************************************************************
   itom measurement system
   URL: http://www.uni-stuttgart.de/ito
   Copyright (C) 2020, Institut fuer Technische Optik (ITO),
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


#include <QtCore/QtPlugin>
#include "itomWidgetsFactory.h"

#include "factories/rangeSliderFactory.h"
#include "factories/brushCreatorButtonFactory.h"
#include "factories/collapsibleGroupBoxFactory.h"
#include "factories/colorPickerButtonFactory.h"
#include "factories/doubleRangeSliderFactory.h"
#include "factories/doubleSliderFactory.h"
#include "factories/doubleSpinBoxFactory.h"
#include "factories/rangeSliderFactory.h"
#include "factories/rangeWidgetFactory.h"
#include "factories/doubleRangeWidgetFactory.h"
#include "factories/sliderWidgetFactory.h"
#include "factories/pathLineEditFactory.h"
#include "factories/popupWidgetFactory.h"
#include "factories/searchBoxFactory.h"
#include "factories/treeComboBoxFactory.h"
#include "factories/menuComboBoxFactory.h"
#include "factories/comboBoxFactory.h"
#include "factories/checkableComboBoxFactory.h"
#include "factories/plotInfoDObjectFactory.h"
#include "factories/plotInfoMarkerFactory.h"
#include "factories/plotInfoPickerFactory.h"
#include "factories/plotInfoShapesFactory.h"
#include "factories/penCreatorButtonFactory.h"
#include "factories/motorAxisControllerFactory.h"
#include "factories/statusLedFactory.h"
#include "factories/fontButtonFactory.h"
#include "factories/paramEditorFactory.h"
#include "factories/pythonLogWidgetFactory.h"
#include "factories/dataObjectTableFactory.h"
#include "factories/dataObjectMetaWidgetFactory.h"

//------------------------------------------------------------------------------------------------
ItomWidgetsFactory::ItomWidgetsFactory(QObject *parent)
    : QObject(parent)
{
    widgets.append(new brushCreatorButtonFactory(this));
    widgets.append(new CollapsibleGroupBoxFactory(this));
    widgets.append(new ColorPickerButtonFactory(this));
    widgets.append(new DoubleRangeSliderFactory(this));
    widgets.append(new DoubleSliderFactory(this));
    widgets.append(new DoubleSpinBoxFactory(this));
    widgets.append(new RangeSliderFactory(this));
    widgets.append(new RangeWidgetFactory(this));
    widgets.append(new DoubleRangeWidgetFactory(this));
    widgets.append(new SliderWidgetFactory(this));
    widgets.append(new PathLineEditFactory(this));
    widgets.append(new PopupWidgetFactory(this));
    widgets.append(new SearchBoxFactory(this));
    widgets.append(new TreeComboBoxFactory(this));
    widgets.append(new MenuComboBoxFactory(this));
    widgets.append(new ComboBoxFactory(this));
    widgets.append(new CheckableComboBoxFactory(this));
    widgets.append(new PenCreatorButtonFactory(this));
    widgets.append(new PlotInfoDObjectFactory(this));
    widgets.append(new PlotInfoMarkerFactory(this));
    widgets.append(new PlotInfoPickerFactory(this));
    widgets.append(new PlotInfoShapesFactory(this));
    widgets.append(new MotorAxisControllerFactory(this));
    widgets.append(new StatusLedFactory(this));
    widgets.append(new FontButtonFactory(this));
    widgets.append(new ParamEditorFactory(this));
    widgets.append(new PythonLogWidgetFactory(this));
    widgets.append(new DataObjectTableFactory(this));
    widgets.append(new DataObjectMetaWidgetFactory(this));
}

//------------------------------------------------------------------------------------------------
QList<QDesignerCustomWidgetInterface*> ItomWidgetsFactory::customWidgets() const
{
    return widgets;
}
