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

#ifndef WIDGETINFOBOX_H
#define WIDGETINFOBOX_H

#include <QtGui>
#include <qwidget.h>

#include "../helper/guiHelper.h"

#include "ui_widgetInfoBox.h"

namespace ito
{

class WidgetInfoBox : public QWidget
{
    Q_OBJECT

public:
    WidgetInfoBox(QString infoText, QWidget *parent = NULL) :
        QWidget(parent)
    {
        ui.setupUi(this);
        ui.lblInfo->setText(infoText);
        QFont f = ui.lblInfo->font();
        float factor = GuiHelper::screenDpiFactor();
        f.setPointSize(factor * f.pointSize());
        ui.lblInfo->setFont(f);

        ui.btnClose->setIcon( QIcon(":/plugins/icons/pluginCloseInstance.png") );
        ui.btnClose->setText("");
        ui.btnClose->setIconSize( QSize(12 * factor, 12 * factor) );

        //setStyleSheet(QString("background-color: %1").arg(QColor(255, 255, 166).name()));
        //setStyleSheet( "QWidget { background-color: blue; }" );
        //this->setPalette( QPalette(Qt::red) );
        setAutoFillBackground(true);
        QPalette pal = this->palette();
        pal.setColor(QPalette::Window, QColor(255, 255, 166));
        this->setPalette(pal);
    }

    ~WidgetInfoBox(){}

    void setInfoText(QString &infoText)
    {
        ui.lblInfo->setText(infoText);
    }

private:
    Ui::WidgetInfoBox ui;

};

} //end namespace ito

#endif
