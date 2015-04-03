/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2015, Institut für Technische Optik (ITO),
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

#include "dialogPipManager.h"

#include <qmessagebox.h>

#include "../global.h"

namespace ito {

DialogPipManager::DialogPipManager(QWidget *parent ) :
    QDialog(parent),
    m_pPipManager(NULL)
{
    ui.setupUi(this);

    m_pPipManager = new PipManager(this);
    connect(m_pPipManager, SIGNAL(pipVersion(QString)), this, SLOT(pipVersion(QString)));
    connect(m_pPipManager, SIGNAL(outputAvailable(QString,bool)), this, SLOT(outputReceived(QString,bool)));

    m_pPipManager->checkPipAvailable();
}

//------------------------------------------------------------
DialogPipManager::~DialogPipManager()
{
    DELETE_AND_SET_NULL(m_pPipManager);
}

//------------------------------------------------------------
void DialogPipManager::pipVersion(const QString &version)
{
    ui.lblPipVersion->setText(version);
}

//------------------------------------------------------------
void DialogPipManager::outputReceived(const QString &text, bool isError)
{
    if (isError)
    {
        logHtml += QString("<p style='color:#ff0000;'>%1</p>").arg(text);
    }
    else
    {
        logHtml += QString("<p style='color:#000000;'>%1</p>").arg(text);
    }
    QString output;
    output = QString("<html><head></head><body style='font-size:8pt; font-weight:400; font-style:normal;'>%1</body></html>").arg(logHtml);
    ui.txtLog->setHtml(output);
}

} //end namespace ito