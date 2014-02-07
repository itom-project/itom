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

#include "dialogOpenFileWithFilter.h"

#include <QtConcurrent/qtconcurrentrun.h>
//#include <qtconcurrentrun.h>
#include <qfileinfo.h>
#include <qfileiconprovider.h>
#include <qmessagebox.h>

namespace ito {

DialogOpenFileWithFilter::DialogOpenFileWithFilter(const QString &filename, const ito::AddInAlgo::FilterDef *filter, QVector<ito::ParamBase> &autoMand, QVector<ito::ParamBase> &autoOut, QVector<ito::Param> &userMand, QVector<ito::Param> &userOpt, ito::RetVal &retValue, QWidget *parent)
    : AbstractFilterDialog(autoMand, autoOut, parent),
    m_pMandParser(NULL),
    m_pOptParser(NULL),
    m_filter(NULL),
    m_filterExecuted(false),
    m_previewMovie(NULL),
    m_acceptedClicked(false)
{
    ui.setupUi(this);

    QString m_filename = filename;
    m_filter = filter;

    QFileInfo info(filename);
    ui.lblFilename->setText( info.fileName() );

    QString var = info.completeBaseName();
    QRegExp regExp("^[a-zA-Z][a-zA-Z0-9_-]*$");
    QRegExpValidator *validator = new QRegExpValidator( regExp, ui.txtPythonVariable );
    ui.txtPythonVariable->setValidator( validator );
    ui.txtPythonVariable->setToolTip( tr("The name must start with a letter followed by numbers or letters [a-z] or [A-Z]") );
    if(regExp.indexIn(var) == -1)
    {
        var.prepend("var");
        if(regExp.indexIn(var) == -1)
        {
            var = "varName";
        }
    }

    ui.txtPythonVariable->setText( var );

    QFileIconProvider *provider = new QFileIconProvider();
    QIcon tempIcon = provider->icon(info);
    ui.lblIcon->setPixmap(tempIcon.pixmap(48,48));
    delete provider;

    ui.lblFilter->setText( filter->m_name );

    QWidget *canvas = new QWidget();
    int curIdx = 0;
    ui.scrollParamsMand->setWidget( canvas );
    m_pMandParser = new ParamInputParser( canvas );

    canvas = new QWidget();
    ui.scrollParamsOpt->setWidget( canvas );
    m_pOptParser = new ParamInputParser( canvas );

    m_pMandParser->createInputMask( userMand );
    m_pOptParser->createInputMask( userOpt );

    m_previewMovie = new QMovie(":/application/icons/loader16x16.gif");
    ui.lblProcessMovie->setMovie( m_previewMovie );
    m_previewMovie->start();
    ui.lblProcessMovie->setVisible(false);
    ui.lblProcessText->setVisible(false);

    connect(&filterCallWatcher, SIGNAL(finished()), this, SLOT(filterCallFinished()));
    

    if(userMand.size() == 0 && userOpt.size() == 0)
    {
        retValue += executeFilter();
        ui.tabWidget->setTabEnabled(0,false);
        ui.tabWidget->setTabEnabled(1,false);
    }
    else
    {
        if(userMand.size() == 0)
        {
            ui.tabWidget->setTabEnabled(0,false);
            curIdx = 1;
        }

        if(userOpt.size() == 0)
        {
            ui.tabWidget->setTabEnabled(1,false);
        }
        ui.tabWidget->setCurrentIndex(curIdx);
    }
}

void DialogOpenFileWithFilter::on_buttonBox_accepted()
{
    ito::RetVal retValue;
    if(ui.txtPythonVariable->text() == "")
    {
        QMessageBox::critical(this, tr("Python variable name missing"), tr("You have to give a variable name, under which the loaded item is saved in the global workspace"));
    }
    else
    {
        if(m_filterExecuted == false)
        {
            m_acceptedClicked = true;
            retValue += executeFilter();
        }
        else
        {
            emit accept();
        }
    }
}

void DialogOpenFileWithFilter::on_tabWidget_currentChanged(int index)
{
    if(index == 0)
    {
        if(m_pMandParser->getItemSize() > 0)
        {
            m_filterExecuted = false;
        }
    }
    else if(index == 1)
    {
        if(m_pOptParser->getItemSize() > 0)
        {
            m_filterExecuted = false;
        }
    }
    else
    {
        if(m_filterExecuted == false)
        {
            on_cmdReload_clicked();
        }
    }
}

void DialogOpenFileWithFilter::on_cmdReload_clicked()
{
    if( filterCall.isRunning() == false)
    {
        ito::RetVal ret = executeFilter();
    }
}



ito::RetVal DialogOpenFileWithFilter::executeFilter()
{
    ito::RetVal retVal;
    QTreeWidgetItem *item = NULL;
    ui.treePreview->clear();
    item = new QTreeWidgetItem();
    item->setData(0,Qt::DisplayRole,tr("loading..."));

    ui.treePreview->addTopLevelItem(item);

    if( m_pMandParser->validateInput(true, retVal, true) == false || m_pOptParser->validateInput(false, retVal, true) == false)
    {
       item->setData(0,Qt::DisplayRole, tr("Invalid parameters."));
    }
    else
    {
        static QVector<ito::ParamBase> paramsMand;
        static QVector<ito::ParamBase> paramsOpt;
        retVal += m_pMandParser->getParameters(paramsMand);
        paramsMand =  m_autoMand + paramsMand;
        retVal += m_pOptParser->getParameters(paramsOpt);
        if(retVal.containsError())
        {
            ui.treePreview->clear();
        }
        else
        {
            ui.cmdReload->setEnabled(false);
            ui.buttonBox->setEnabled(false);
            ui.lblProcessMovie->setVisible(true);
            ui.lblProcessText->setVisible(true);

            filterCall = QtConcurrent::run(m_filter->m_filterFunc, &paramsMand, &paramsOpt, &m_autoOut);
            filterCallWatcher.setFuture(filterCall);
        }
    }

    if(retVal.containsError())
    {
        QString text = tr("An error occurred while loading the file.");
        if(retVal.errorMessage()) text.append( "\n" ).append(retVal.errorMessage());
        QMessageBox::critical( this, tr("Error while loading file"), text);
    }
    return retVal;
}

void DialogOpenFileWithFilter::filterCallFinished()
{
    ito::RetVal retValue =  filterCall.result();
    ui.lblProcessMovie->setVisible(false);
    ui.lblProcessText->setVisible(false);

    if(retValue.containsError())
    {
        ui.treePreview->clear();
        QString text = tr("An error occurred while loading the file.");
        if(retValue.errorMessage()) text.append( "\n" ).append(retValue.errorMessage());
        QMessageBox::critical( this, tr("Error while loading file"), text);
        m_acceptedClicked = false;

        if(m_pMandParser->getItemSize() == 0 && m_pOptParser->getItemSize() == 0)
        {
            //the user cannot do anything, therefore close the dialog
            emit reject();
        }
    }
    else
    {
        m_filterExecuted = true;
        QList<QTreeWidgetItem*> items = renderAutoMandAndOutResult();
        ui.treePreview->clear();
        ui.treePreview->addTopLevelItems(items);
        ui.treePreview->expandAll();

        if(m_acceptedClicked)
        {
            emit accept();
        }
    }

    ui.cmdReload->setEnabled(true);
    ui.buttonBox->setEnabled(true);
}

void DialogOpenFileWithFilter::closeEvent(QCloseEvent *e)
{
    if(filterCall.isRunning())
    {
        QMessageBox::critical(this, tr("Procedure still running"), tr("The file is still being loaded. Please wait..."));
        e->ignore();
    }
}

} //end namespace ito
