/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2023, Institut fuer Technische Optik (ITO),
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

    --------------------------------
    This class is a modified version of the class QToolTip of the
    Qt framework (licensed under LGPL):
    https://code.woboq.org/qt5/qtbase/src/widgets/kernel/qtooltip.cpp.html
*********************************************************************** */

#include "pyCodeReferenceRenamer.h"
#include "../python/pythonEngine.h"
#include "../AppManagement.h"
#include "../python/pythonJedi.h"

#include <qlistwidget.h>
#include <qdialog.h>
#include <qlayout.h>
#include <qdialogbuttonbox.h>

namespace ito {

//-------------------------------------------------------------------------------------
PyCodeReferenceRenamer::PyCodeReferenceRenamer(QObject* parent /*= nullptr*/) :
    QObject(parent)
{
    m_pPythonEngine = AppManagement::getPythonEngine();
}

//-------------------------------------------------------------------------------------
PyCodeReferenceRenamer::~PyCodeReferenceRenamer()
{
}

//-------------------------------------------------------------------------------------
void PyCodeReferenceRenamer::rename(const int &line, const int &column, const QString &fileName, const QString &newName)
{
    ito::JediRenameRequest request;
    request.m_code = "";
    request.m_callbackFctName = "onJediRenameResultAvailable";
    request.m_col = column;
    request.m_line = line;
    request.m_fileName = fileName;
    request.m_newName = newName;
    request.m_sender = this;
    PythonEngine* pyEng = (PythonEngine*)m_pPythonEngine;
    pyEng->enqueueJediRenameRequest(request);
}

//-------------------------------------------------------------------
void PyCodeReferenceRenamer::onJediRenameResultAvailable(QVector<ito::JediRename> fileToChange)
{
    QDialog* dialog = new QDialog();
    dialog->setWindowTitle("List of files to change references");

    QListWidget* listWidget = new QListWidget;
    listWidget->setAlternatingRowColors(true);

    QVectorIterator<JediRename> iter(fileToChange);
    foreach (const JediRename &file, fileToChange)
    {
        QListWidgetItem* item = new QListWidgetItem(file.m_flePath);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(Qt::Checked);
        listWidget->addItem(item);
    }

    QDialogButtonBox* buttonBox =
        new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);


    QVBoxLayout* layout = new QVBoxLayout(dialog);
    layout->addWidget(listWidget);
    layout->addWidget(buttonBox);

    dialog->setLayout(layout);
    dialog->show();
}

} // namespace ito
