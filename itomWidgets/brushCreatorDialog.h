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
#ifndef BRUSHCREATORDIALOG_H
#define BRUSHCREATORDIALOG_H

#include"ui_brushCreatorDialog.h"
#include<QDialog>
#include<qbrush.h>

class BrushCreatorDialog : public QDialog
{
    Q_OBJECT

public:
    BrushCreatorDialog(QBrush &inputBrush, QWidget *parent = NULL);
    ~BrushCreatorDialog();
    void setBrush(const QBrush &brush);
    void setShowAlphaChannel(bool showAlphaChannel);
    void synchronizeGUI();
    QBrush getBrush();

private:
    Ui::brushCreatorDialog ui;
    QBrush &brush;

    void updateBrush();
private slots:
    void on_buttonBox_clicked(QAbstractButton* btn);

};
#endif
