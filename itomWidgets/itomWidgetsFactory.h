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

#ifndef ITOMWIDGETSFACTORY_H
#define ITOMWIDGETSFACTORY_H

#include "qglobal.h"
#include <QtUiPlugin/QDesignerCustomWidgetCollectionInterface>

class ItomWidgetsFactory : public QObject, public QDesignerCustomWidgetCollectionInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "org.qt-project.Qt.QDesignerCustomWidgetCollectionInterface" )
    Q_INTERFACES(QDesignerCustomWidgetCollectionInterface)

public:
     ItomWidgetsFactory(QObject *parent = 0);

     virtual QList<QDesignerCustomWidgetInterface*> customWidgets() const;

 private:
     QList<QDesignerCustomWidgetInterface*> widgets;
};

#endif // ITOMWIDGETSFACTORY_H
