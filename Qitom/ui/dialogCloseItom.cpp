/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#include "dialogCloseItom.h"

#include "../AppManagement.h"

#include <QTimerEvent>
#include <qmovie.h>

namespace ito
{

DialogCloseItom::DialogCloseItom(QWidget *parent) :
    QDialog(parent),
	m_secondsToWait(20),
	m_secondsElapsed(0),
	m_timerID(-1)
{
    ui.setupUi(this);
	
    setWindowModality(Qt::WindowModal);
	ui.pushButtonOk->setFocus();

	ui.progressBarClose->setValue(0);
	ui.progressBarClose->setMaximum(m_secondsToWait);
	ui.progressBarClose->setVisible(false);
	QPixmap pixmap(":/script/icons/stopScript.png");
	ui.labelImage->setPixmap(pixmap.scaled(32,32, Qt::KeepAspectRatio));

}

void DialogCloseItom::on_pushButtonOk_clicked()
{
	PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

	if (pyEngine)
	{

		//QPixmap pixmap(":/script/icons/loader32x32trans.gif");
		QMovie *loadGif = new QMovie(":/application/icons/loader32x32trans.gif");

		ui.labelImage->setMovie(loadGif);
		loadGif->start();
		
		pyEngine->pythonInterruptExecution(); // close python
		
		ui.pushButtonOk->setEnabled(false);
		ui.progressBarClose->setVisible(true);
		m_secondsElapsed = 0.0;
		m_timerID = startTimer(500);

	}	

}

void DialogCloseItom::on_pushButtonCancel_clicked()
{
	reject();
}

void DialogCloseItom::timerEvent(QTimerEvent *event)
{
	m_secondsElapsed += 0.5;
	
	ui.progressBarClose->setValue(ui.progressBarClose->maximum() * m_secondsElapsed / (float)m_secondsToWait);
	PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
	if (!pyEngine->isPythonBusy())
	{
		accept();
	}

	if (m_secondsElapsed >= m_secondsToWait)
	{
		//do something, message box...
		killTimer(m_timerID);
		reject();
	}
}

} //end namespace ito