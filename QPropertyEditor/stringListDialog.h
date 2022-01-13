/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2021, Institut fuer Technische Optik (ITO),
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

#ifndef STRINGLISTDIALOG_H
#define STRINGLISTDIALOG_H

#include "ui_stringListDialog.h"

#include <qdialog.h>

class QListWidgetItem;

class StringListDialog : public QDialog
{
    Q_OBJECT

public:
    explicit StringListDialog(const QStringList& stringList, QWidget* parent);

    QListWidget* listWidget() const
    {
        return ui.listWidget;
    }
    void setNewItemText(const QString& tpl)
    {
        m_newItemText = tpl;
    }
    QString newItemText() const
    {
        return m_newItemText;
    }
    void setCurrentIndex(int idx);
    QStringList getStringList();

private slots:
    void on_newListItemButton_clicked();
    void on_deleteListItemButton_clicked();
    void on_moveListItemUpButton_clicked();
    void on_moveListItemDownButton_clicked();
    void on_listWidget_currentRowChanged();
    void on_listWidget_itemDoubleClicked(QListWidgetItem* item);

protected:
    virtual void setItemData(int role, const QVariant& v);
    virtual QVariant getItemData(int role) const;

private:
    void updateEditor();
    Ui::StringListDialog ui;
    bool m_updating;
    QString m_newItemText;
};

#endif // STRINGLISTDIALOG_H
