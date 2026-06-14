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

#include "dialogCloseItom.h"

#include "../AppManagement.h"
#include "../helper/guiHelper.h"

#include <QTimerEvent>
#include <QCloseEvent>
#include <qmovie.h>

namespace ito
{

    //------------------------------------------------------------------------------------------------------
DialogCloseItom::DialogCloseItom(QWidget *parent) :
    QDialog(parent),
	m_secondsToWait(20),
	m_secondsElapsed(0),
	m_timerID(-1)
{
    ui.setupUi(this);

    setWindowModality(Qt::WindowModal);
    ui.btnInterrupt->setDefault(true);

	ui.progressBarClose->setValue(0);
	ui.progressBarClose->setMaximum(m_secondsToWait);
	ui.progressBarClose->setVisible(false);
	QPixmap pixmap(":/script/icons/stopScript.png");
    float dpiFactor = GuiHelper::screenDpiFactor(); //factor related to 96dpi (1.0)
    ui.labelImage->setPixmap(pixmap.scaled(32.0 / dpiFactor, 32.0 / dpiFactor, Qt::KeepAspectRatio));

}

//------------------------------------------------------------------------------------------------------
void DialogCloseItom::on_btnInterrupt_clicked()
{
	PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());

	if (pyEngine)
	{
		QMovie *loadGif = new QMovie(":/application/icons/loader32x32trans.gif");

		ui.labelImage->setMovie(loadGif);
		loadGif->start();

        ui.lbl2->setText(tr("Try to interrupt Python..."));

		pyEngine->pythonInterruptExecutionThreadSafe(); // close python

        ui.btnInterrupt->setEnabled(false);
        ui.btnCancel->setEnabled(false);
        ui.progressBarClose->setMaximum(m_secondsToWait);
        ui.progressBarClose->setValue(0);
		ui.progressBarClose->setVisible(true);

		m_secondsElapsed = 0.0;
		m_timerID = startTimer(1000);
	}

}

//------------------------------------------------------------------------------------------------------
void DialogCloseItom::on_btnCancel_clicked()
{
	reject();
}

//------------------------------------------------------------------------------------------------------
void DialogCloseItom::timerEvent(QTimerEvent *event)
{
	m_secondsElapsed += 1;

	ui.progressBarClose->setValue(m_secondsElapsed);
	PythonEngine *pyEngine = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
	if (!pyEngine->isPythonBusy())
	{
		accept();
	}

	if (m_secondsElapsed >= m_secondsToWait)
	{
		//do something, message box...
		killTimer(m_timerID);

        ui.btnInterrupt->setEnabled(true);
        ui.btnCancel->setEnabled(true);
        ui.progressBarClose->setVisible(false);

        ui.lbl2->setText(tr("Python did not stop. Do you want to retry to interrupt Python?"));
	}
}

//------------------------------------------------------------------------------------------------------
void DialogCloseItom::closeEvent(QCloseEvent *event)
{
    //this event is called if the user tries to close the dialog by the close-button in the title bar.
    //Ignore this event if the python interruption process is currently running.

    if (ui.progressBarClose->isVisible())
    {
        event->ignore();
    }
    else
    {
        event->accept();
    }
}

//----------------------------------------------------------------------------------------
void DialogCloseItom::keyPressEvent(QKeyEvent *event)
{
	int key = event->key();
	//Qt::KeyboardModifiers modifiers = event->modifiers();

	if (key == Qt::Key_Escape)
	{
		reject();
	}

}

} //end namespace ito
