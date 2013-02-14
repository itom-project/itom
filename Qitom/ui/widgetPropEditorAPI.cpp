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



WidgetPropEditorAPI::WidgetPropEditorAPI(QWidget *parent) :
    AbstractPropertyPageWidget(parent)
{
    ui.setupUi(this);

    QDir canonicalBase = QCoreApplication::applicationDirPath();
    m_canonicalBasePath = canonicalBase.canonicalPath();
    ui.lblBasePath->setText(tr("base path for relative pathes: ") + m_canonicalBasePath);
    ui.btnAdd->setEnabled(true);
    ui.btnRemove->setEnabled(false);

    ui.line->setVisible(false);
    ui.btnPrepareAPI->setVisible(false);

    m_pApiManager = ito::QsciApiManager::getInstance();

    m_lastApiFileDirectory = QDir::cleanPath(QCoreApplication::applicationDirPath());

    connect(m_pApiManager->getQsciAPIs(), SIGNAL(apiPreparationFinished()), this, SLOT(apiPreparationFinished()));
    connect(m_pApiManager->getQsciAPIs(), SIGNAL(apiPreparationCancelled()), this, SLOT(apiPreparationCancelled()));
    connect(m_pApiManager->getQsciAPIs(), SIGNAL(apiPreparationStarted()), this, SLOT(apiPreparationStarted()));
}

WidgetPropEditorAPI::~WidgetPropEditorAPI()
{
    m_pApiManager = NULL;
}

void WidgetPropEditorAPI::readSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    settings.beginGroup("PyScintilla");
    QString filename;

    int size = settings.beginReadArray("apiFiles");
    for (int i = 0; i < size; ++i) 
    {
        settings.setArrayIndex(i);

        QFileInfo fileInfo(settings.value("file",QString()).toString());
        if(fileInfo.exists())
        {
            filename = fileInfo.canonicalFilePath();
            if(filename.startsWith( m_canonicalBasePath ))
            {
                filename = filename.mid( m_canonicalBasePath.length() );
            }

            ui.listWidget->addItem(filename);    
        }
    }

    settings.endArray();


    settings.endGroup();
}

void WidgetPropEditorAPI::writeSettings()
{
    QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
    QStringList files;
    settings.beginGroup("PyScintilla");
    QString filename;

    settings.beginWriteArray("apiFiles");
    for(int i = 0 ; i < ui.listWidget->count() ; i++)
    {
        settings.setArrayIndex(i);
        filename = ui.listWidget->item(i)->text();
        if(filename.contains(":") == false)
        {
            filename.prepend( m_canonicalBasePath );
        }
        settings.setValue("file", filename);
        files.append(filename);
    }

    settings.endArray();
    settings.endGroup();

    m_pApiManager->updateAPI(files);
}

void WidgetPropEditorAPI::on_listWidget_currentItemChanged(QListWidgetItem* current, QListWidgetItem* /*previous*/)
{
    ui.btnRemove->setEnabled(current != NULL);
}

void WidgetPropEditorAPI::on_btnAdd_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, "load python api file",m_lastApiFileDirectory,"python api file (*.api)");
     
    if(!filename.isEmpty())
    {
        m_lastApiFileDirectory = QDir::cleanPath(QFileInfo(filename).path());
        QFileInfo fileInfo(filename);
        filename = fileInfo.canonicalFilePath();
        if(filename.startsWith( m_canonicalBasePath ))
        {
            filename = filename.mid( m_canonicalBasePath.length() );
        }

        ui.listWidget->addItem(filename);
    }
}

void WidgetPropEditorAPI::on_btnRemove_clicked()
{
    if(ui.listWidget->currentItem())
    {
        ui.listWidget->takeItem( ui.listWidget->currentIndex().row());
    }
}

void WidgetPropEditorAPI::on_listWidget_itemActivated(QListWidgetItem* item)
{
    if(item)
    {
        item->setFlags(item->flags() | Qt::ItemIsEditable);
        ui.listWidget->editItem(item);
    }
}

void WidgetPropEditorAPI::apiPreparationFinished()
{
    ui.btnPrepareAPI->setText(tr("generate lookup table by API files"));
    QMessageBox msgBox(this);
    msgBox.setText(tr("The API generation has been finished"));
    msgBox.exec();
}

void WidgetPropEditorAPI::apiPreparationCancelled()
{
    ui.btnPrepareAPI->setText(tr("generate lookup table by API files"));
}

void WidgetPropEditorAPI::apiPreparationStarted()
{
    ui.btnPrepareAPI->setText(tr("cancel preparation"));
}

void WidgetPropEditorAPI::on_btnPrepareAPI_clicked()
{
    if(m_pApiManager->isPreparing())
    {
        m_pApiManager->getQsciAPIs()->cancelPreparation();
    }
    else
    {
        QStringList files;
        
        for(int i = 0 ; i < ui.listWidget->count() ; i++)
        {
            files.append(ui.listWidget->item(i)->text());
        }

        int ret = m_pApiManager->updateAPI(files);

        if(ret == 1)
        {
            QMessageBox msgBox(this);
            msgBox.setText(tr("API files are already up-to-date"));
            msgBox.exec();
        }
    }
}