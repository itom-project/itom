/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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

#ifndef WIDGETPROPEDITORAPI_H
#define WIDGETPROPEDITORAPI_H

#include "abstractPropertyPageWidget.h"

#include <qlistwidget.h>
#include <qwidget.h>
#include <qstring.h>

#include "../organizer/qsciApiManager.h"

#include "ui_widgetPropEditorAPI.h"

namespace ito
{

class WidgetPropEditorAPI : public AbstractPropertyPageWidget
{
    Q_OBJECT

public:
    WidgetPropEditorAPI(QWidget *parent = NULL);
    ~WidgetPropEditorAPI();

    void readSettings();
    void writeSettings();

protected:

private:
    Ui::WidgetPropEditorAPI ui;

    ito::QsciApiManager *m_pApiManager;

    QString m_lastApiFileDirectory;
    QString m_notExistAppendix;

    QString m_canonicalBasePath;

    bool m_changes;

private slots:
    void on_listWidget_currentItemChanged(QListWidgetItem* current, QListWidgetItem* previous);
    void on_listWidget_itemActivated(QListWidgetItem* item);
    void on_btnAdd_clicked();
    void on_btnRemove_clicked();

};

} //end namespace ito

#endif