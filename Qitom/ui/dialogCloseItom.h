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

#ifndef DIALOGCLOSEITOM_H
#define DIALOGCLOSEITOM_H

#include "../python/pythonEngine.h"

#include <qdialog.h>
#include <qstring.h>

#include "ui_dialogCloseItom.h"

class QTimerEvent;

namespace ito
{

class DialogCloseItom : public QDialog
{
    Q_OBJECT

public:
    DialogCloseItom(QWidget *parent = 0);
    ~DialogCloseItom() {}

protected:
	void timerEvent(QTimerEvent *event);
    void closeEvent(QCloseEvent *event);
	void keyPressEvent(QKeyEvent *event);

private:
    Ui::DialogCloseItom ui;

	int m_secondsToWait;
	int m_secondsElapsed;
	int m_timerID;

private slots:
    void on_btnInterrupt_clicked();
    void on_btnCancel_clicked();

};

} //end namespace ito

#endif
