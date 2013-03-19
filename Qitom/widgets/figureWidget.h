/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#ifndef FIGUREWIDGET_H
#define FIGUREWIDGET_H

#include "abstractDockWidget.h"

namespace ito {

class FigureWidget : public AbstractDockWidget
{
    Q_OBJECT
public:
    FigureWidget(const QString &title, bool docked, bool isDockAvailable, QWidget *parent = 0, Qt::WindowFlags flags = 0);
    ~FigureWidget();

protected:
    
    void createActions();
    void createMenus();
    void createToolBars();
    void createStatusBar();
    void updateActions();
    void updatePythonActions(){ updateActions(); }

    void closeEvent(QCloseEvent *event) {};

private:


signals:
    
private slots:
    
};

} //end namespace ito

#endif
