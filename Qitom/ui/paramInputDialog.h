/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#include "ui_paramInputDialog.h"

#include "common/paramMeta.h"

#include <qdialog.h>
#include <qStyledItemDelegate.h>

//-------------------------------------------------------------------------------------
class QListWidgetItem;

namespace ito {

enum tParamType
{
    none        = 0x0,
    intArray    = 0x1,
    doubleArray = 0x2,
    charArray   = 0x4
};

//-------------------------------------------------------------------------------------
class LineEditDelegate : public QStyledItemDelegate
{
    Q_OBJECT

public:
    explicit LineEditDelegate(const double minVal, const double maxVal, const tParamType paramType, QObject *parent = 0);

    QWidget* createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const;

    void setEditorData(QWidget *editor, const QModelIndex &index) const;
    void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;
    void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const;

private:
    ito::ParamMeta m_meta;
    double m_minVal;
    double m_maxVal;
    tParamType m_paramType;
};

//-------------------------------------------------------------------------------------
class ParamInputDialog: public QDialog
{
    Q_OBJECT

public:
    explicit ParamInputDialog(const QStringList &stringList, const ito::ParamMeta *meta, const tParamType paramType, QWidget *parent);
    ~ParamInputDialog();

    QListWidget *listWidget() const { return ui.listWidget; }
    void setNewItemText(const QString &tpl) { m_newItemText = tpl; }
    QString newItemText() const { return m_newItemText; }
    void setCurrentIndex(int idx);
    QStringList getStringList();
    QRegExp m_RegExp;
    LineEditDelegate *m_lineEditDel;

private slots:
    void on_newListItemButton_clicked();
    void on_deleteListItemButton_clicked();
    void on_moveListItemUpButton_clicked();
    void on_moveListItemDownButton_clicked();
    void on_listWidget_currentRowChanged();
    void on_listWidget_itemDoubleClicked(QListWidgetItem *item);
    void on_buttonBox_clicked(QAbstractButton* btn);

protected:
    virtual void setItemData(int role, const QVariant &v);
    virtual QVariant getItemData(int role) const;

private:
    void updateEditor();
    Ui::paramInputDialog ui;
    bool m_updating;
    QString m_newItemText;
    int m_minSize;
    int m_maxSize;
    int m_stepSize;
    double m_minVal;
    double m_maxVal;
};

} //end namespace ito

#endif // STRINGLISTDIALOG_H
