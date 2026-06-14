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

#ifndef DIALOGGOTO_H
#define DIALOGGOTO_H

#include <qdialog.h>
#include <qstring.h>

#include "ui_dialogGoto.h"
#include "itomSpinBox.h"

namespace ito
{

class DialogGoto : public QDialog
{
    Q_OBJECT

public:
    DialogGoto(const int maxLine, const int curLine, const int maxChar, const int curChar, QWidget *parent = 0);
    ~DialogGoto() {}

    void getData(bool &lineNotChar, int &curValue);

private:
    Ui::DialogGoto ui;

    QString m_captionLine;
    QString m_captionChar;
    int m_maxLine;
    int m_maxChar;

private slots:
    void on_radioLine_clicked();
    void on_radioChar_clicked();

};

} //end namespace ito

#endif
