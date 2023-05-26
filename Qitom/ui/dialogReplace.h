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

#ifndef DIALOGREPLACE_H
#define DIALOGREPLACE_H

#include <QtGui>
#include <qdialog.h>

#include "ui_dialogReplace.h"
#include "itomSpinBox.h"

class QCompleter; //forward declaration

namespace ito
{

class DialogReplace : public QDialog
{
    Q_OBJECT

public:
    DialogReplace(QWidget *parent = 0);
    ~DialogReplace() {}

//    void setData(const QString &defaultText, const int &lineFrom, const int &indexFrom, const int &lineTo, const int &indexTo);
    void setData(const QString &defaultText, const bool &rowSelected);

protected:
    virtual void closeEvent(QCloseEvent *event);

private:
    Ui::DialogReplace ui;

    void comboBoxAddItem(const QString &text, QComboBox *comboBox);
    int comboBoxGetIndex(const QString &text, QComboBox *comboBox) const;
    void setRegularMode(const bool isRegularMode);

    QCompleter *m_pCompleter;

private slots:
    void on_pushButtonFindNext_clicked();
    void on_pushButtonReplace_clicked();
    void on_pushButtonReplaceAll_clicked();
    void on_pushButtonExpand_clicked();
    void on_checkBoxReplaceWith_clicked();

public slots:
    void userCursorPositionChanged() {};

signals:
    void findNext(QString expr, bool regExpr, bool caseSensitive, bool wholeWord, bool wrap, bool forward = true, bool isQuickSeach = false);
    void replaceSelection(QString expr, QString replace);
    void replaceAll(QString expr, QString replace, bool regExpr, bool caseSensitive, bool wholeWord, bool findInSel);
};

} //end namespace ito

#endif
