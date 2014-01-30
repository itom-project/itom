/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
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

#include "widgetPropEditorAPI.h"

#include "../global.h"
#include "../AppManagement.h"


#include <qsettings.h>
#include <qcoreapplication.h>
#include <qfiledialog.h>
#include <qstringlist.h>
#include <qmessagebox.h>

#include <qcoreapplication.h>
#include <qdir.h>

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorAPI::WidgetPropEditorAPI(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);

    QDir canonicalBase = QCoreApplication::applicationDirPath();
    m_canonicalBasePath = canonicalBase.canonicalPath();
    ui.lblBasePath->setText(tr("base path for relative pathes: ") + m_canonicalBasePath);
    ui.btnAdd->setEnabled(true);
    ui.btnRemove->setEnabled(false);

    m_notExistAppendix = tr("[does not exist]");

    m_pApiManager = ito::QsciApiManager::getInstance();

    m_lastApiFileDirectory = QDir::cleanPath(QCoreApplication::applicationDirPath());

    m_changes = false;

    /*connect(m_pApiManager->getQsciAPIs(), SIGNAL(apiPreparationFinished()), this, SLOT(apiPreparationFinished()));
    connect(m_pApiManager->getQsciAPIs(), SIGNAL(apiPreparationCancelled()), this, SLOT(apiPreparationCancelled()));
    connect(m_pApiManager->getQsciAPIs(), SIGNAL(apiPreparationStarted()), this, SLOT(apiPreparationStarted()));*/
}

//----------------------------------------------------------------------------------------------------------------------------------
WidgetPropEditorAPI::~WidgetPropEditorAPI()
{
    m_pApiManager = NULL;
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorAPI::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("PyScintilla");
    QString filename;
    QDir baseDir(m_canonicalBasePath);

    int size = settings.beginReadArray("apiFiles");
    for (int i = 0; i < size; ++i) 
    {
        settings.setArrayIndex(i);

        filename = settings.value("file", QString()).toString();
        filename = baseDir.absoluteFilePath(filename);
        QFileInfo fileInfo(filename);
        if (fileInfo.exists())
        {
            filename = baseDir.relativeFilePath(filename);
            ui.listWidget->addItem(filename);    
        }
        else
        {
            filename = baseDir.relativeFilePath(filename);
            ui.listWidget->addItem(filename + " " + m_notExistAppendix);
        }
    }

    settings.endArray();
    settings.endGroup();
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorAPI::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    QStringList files;
    settings.beginGroup("PyScintilla");
    QString filename;
    QDir baseDir(m_canonicalBasePath);

    settings.beginWriteArray("apiFiles");
    for (int i = 0 ; i < ui.listWidget->count() ; i++)
    {
        settings.setArrayIndex(i);
        filename = ui.listWidget->item(i)->text();
        if (filename.endsWith(m_notExistAppendix))
        {
            filename = filename.left(filename.length() - m_notExistAppendix.length()).trimmed();
        }
        filename = baseDir.relativeFilePath(filename); //save relative filenames
        settings.setValue("file", filename);
        files.append(filename);
    }

    settings.endArray();
    settings.endGroup();

    if (m_changes)
    {
        m_pApiManager->updateAPI(files);
        m_changes = false;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorAPI::on_listWidget_currentItemChanged(QListWidgetItem* current, QListWidgetItem* /*previous*/)
{
    ui.btnRemove->setEnabled(current != NULL);
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorAPI::on_btnAdd_clicked()
{
    QStringList filenames = QFileDialog::getOpenFileNames(this, tr("load python api file"), m_lastApiFileDirectory, tr("python api file (*.api)"));
    QDir baseDir(m_canonicalBasePath);
     
    foreach (QString filename, filenames)
    {
        m_changes = true;

        m_lastApiFileDirectory = QDir::cleanPath(QFileInfo(filename).path());
        QFileInfo fileInfo(filename);
        filename = fileInfo.canonicalFilePath();
        filename = baseDir.relativeFilePath(filename); //get relative filenames with respect to application directory
        /*if (filename.startsWith(m_canonicalBasePath))
        {
            filename = filename.mid(m_canonicalBasePath.length());
        }*/

        if (ui.listWidget->findItems(filename, Qt::MatchExactly).isEmpty())
        {
            ui.listWidget->addItem(filename);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorAPI::on_btnRemove_clicked()
{
    qDeleteAll(ui.listWidget->selectedItems());
    m_changes = true;
}

//----------------------------------------------------------------------------------------------------------------------------------
void WidgetPropEditorAPI::on_listWidget_itemActivated(QListWidgetItem* item)
{
    if (item)
    {
        //item->setFlags(item->flags() | Qt::ItemIsEditable);
        //ui.listWidget->editItem(item);
    }
}