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

#include "dialogOpenFileWithFilter.h"

#include "../AppManagement.h"
#include "../helper/guiHelper.h"

#include <QtConcurrent/qtconcurrentrun.h>
#include <qfileinfo.h>
#include <qfileiconprovider.h>
#include <qmessagebox.h>
#include <qregularexpression.h>

namespace ito {

//------------------------------------------------------------------------------------------------------------
    DialogOpenFileWithFilter::DialogOpenFileWithFilter(const QString &filename, const ito::AddInAlgo::FilterDef *filter, QVector<ito::ParamBase> &autoMand, QVector<ito::ParamBase> &autoOut, QVector<ito::Param> &userMand, QVector<ito::Param> &userOpt, ito::RetVal &retValue, CheckVarname varnameCheck /*= CheckNo*/, QWidget *parent /*= NULL*/)
    : AbstractFilterDialog(autoMand, autoOut, parent),
    m_pMandParser(NULL),
    m_pOptParser(NULL),
    m_filter(NULL),
    m_filterExecuted(false),
    m_previewMovie(NULL),
    m_acceptedClicked(false),
    m_checkVarname(varnameCheck)
{
    ui.setupUi(this);

    QString m_filename = filename;
    m_filter = filter;

    QFileInfo info(filename);
    ui.lblFilename->setText( info.fileName() );

    QString var = info.completeBaseName();
    QRegularExpression regExp("^[a-zA-Z][a-zA-Z0-9_]*$");
    auto *validator = new QRegularExpressionValidator( regExp, ui.txtPythonVariable );
    ui.txtPythonVariable->setValidator( validator );
    ui.txtPythonVariable->setToolTip( tr("The name must start with a letter followed by numbers or letters [a-z] or [A-Z]") );


    if(var.indexOf(regExp) == -1)
    {
        var.prepend("var");
        var.replace("-", "_");

        if(var.indexOf(regExp) == -1)
        {
            var = "varName";
        }
    }

    ui.txtPythonVariable->setText( var );

    float dpiFactor = GuiHelper::screenDpiFactor(); //factor related to 96dpi (1.0)

    QFileIconProvider *provider = new QFileIconProvider();
    QIcon tempIcon = provider->icon(info);
	ui.lblIcon->setPixmap(tempIcon.pixmap(dpiFactor * 48, dpiFactor * 48));
	ui.lblIcon->setMaximumSize(dpiFactor * 48, dpiFactor * 48);
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

    tempIcon = QIcon(":/plugins/icons/sendToPython.png");
    ui.lblImage->setPixmap(tempIcon.pixmap(16 * dpiFactor, 16 * dpiFactor));

    activateWindow(); //let the import dialog be shown on top even if a file is currently dropped from another explorer window...
}

//------------------------------------------------------------------------------------------------------------
void DialogOpenFileWithFilter::on_buttonBox_accepted()
{
    ito::RetVal retValue;
    QObject *pyEng = static_cast<QObject*>(AppManagement::getPythonEngine());
    bool success = true;

    if(ui.txtPythonVariable->text() == "")
    {
        QMessageBox::critical(this, tr("Python variable name missing"), tr("You have to give a variable name, under which the loaded item is saved in the global workspace"));
        success = false;
    }
    else
    {
        if (m_checkVarname != CheckNo)
        {
            bool globalNotLocal = (m_checkVarname == CheckGlobalWorkspace) ? true : false;
            QStringList pythonVarNames;
            pythonVarNames << ui.txtPythonVariable->text();
            QSharedPointer<IntList> existing(new IntList());
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());

            QMetaObject::invokeMethod(pyEng, "checkVarnamesInWorkspace", Q_ARG(bool, globalNotLocal), Q_ARG(QStringList, pythonVarNames), Q_ARG(QSharedPointer<IntList>, existing), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
            if (locker.getSemaphore()->wait(5000))
            {
                if ((*existing)[0] == 1)
                {
                    QMessageBox::StandardButton result = QMessageBox::question(this, tr("Python variable name already exists"), tr("The variable name %1 already exists in this workspace. Do you want to overwrite it?").arg(ui.txtPythonVariable->text()), QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel, QMessageBox::No);
                    if (result == QMessageBox::No || result == QMessageBox::Cancel)
                    {
                        success = false;
                    }
                }
                else if ((*existing)[0] == 2)
                {
                    QMessageBox::critical(this, tr("Python variable name already exists"), tr("The variable name %1 already exists in this workspace. It cannot be overwritten since it is a function, method, type or class. Choose a new name.").arg(ui.txtPythonVariable->text()));
                    success = false;
                }
            }
            else
            {
                QMessageBox::critical(this, tr("Timeout while verifiying variable name"), tr("A timeout occurred while checking for the existence of the variable name in Python. Please try it again."));
                success = false;
            }
        }
    }

    if (success)
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

//------------------------------------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------------------------------------
void DialogOpenFileWithFilter::on_cmdReload_clicked()
{
    if( filterCall.isRunning() == false)
    {
        ito::RetVal ret = executeFilter();
    }
}


//------------------------------------------------------------------------------------------------------------
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
        m_paramsMand.clear();
        m_paramsOpt.clear();
        retVal += m_pMandParser->getParameters(m_paramsMand);
        m_paramsMand = m_autoMand + m_paramsMand;
        retVal += m_pOptParser->getParameters(m_paramsOpt);
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
            ui.groupPython->setEnabled(false);
            ui.scrollParamsMand->setEnabled(false);
            ui.scrollParamsOpt->setEnabled(false);

            QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
            QApplication::processEvents(QEventLoop::ExcludeSocketNotifiers); //the WaitCursor only becomes visible if the event loop of the main thread is called once.
                                                                             //(it is not allowed to filter  QEventLoop::ExcludeUserInputEvents here out, since mouse events
                                                                             //have to be passed to the operating system. Else the cursor is not changed. - at least with Windows)

            const ito::AddInAlgo::FilterDefExt *filterExt = dynamic_cast<const ito::AddInAlgo::FilterDefExt*>(m_filter);

            if (filterExt)
            {
                QSharedPointer<ito::FunctionCancellationAndObserver> emptyObserver; //no observer initialized
                //starts loading the file in another thread. If this is done, filterCallFinished is executed
                filterCall = QtConcurrent::run(filterExt->m_filterFuncExt, &m_paramsMand, &m_paramsOpt, &m_autoOut, emptyObserver);
            }
            else
            {
                //starts loading the file in another thread. If this is done, filterCallFinished is executed
                filterCall = QtConcurrent::run(m_filter->m_filterFunc, &m_paramsMand, &m_paramsOpt, &m_autoOut);
            }

            filterCallWatcher.setFuture(filterCall);
        }
    }

    if(retVal.containsError())
    {
        QString text = tr("An error occurred while loading the file.");
        if(retVal.hasErrorMessage()) text.append( "\n" ).append(QLatin1String(retVal.errorMessage()));
        QMessageBox::critical( this, tr("Error while loading file"), text);
    }
    else if (retVal.containsWarning())
    {
        QString text = tr("A warning occurred while loading the file.");
        if(retVal.hasErrorMessage()) text.append( "\n" ).append(QLatin1String(retVal.errorMessage()));
        QMessageBox::warning( this, tr("Warning while loading file"), text);
    }

    return retVal;
}

//------------------------------------------------------------------------------------------------------------
void DialogOpenFileWithFilter::filterCallFinished()
{
    ito::RetVal retValue =  filterCall.result();
    ui.lblProcessMovie->setVisible(false);
    ui.lblProcessText->setVisible(false);
    QApplication::restoreOverrideCursor();

    if(retValue.containsError())
    {
        ui.treePreview->clear();
        QString text = tr("An error occurred while loading the file.");
        if(retValue.hasErrorMessage()) text.append( "\n" ).append(QLatin1String(retValue.errorMessage()));
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
        if (retValue.containsWarning())
        {
            QString text = tr("A warning occurred while loading the file.");
            if(retValue.hasErrorMessage()) text.append( "\n" ).append(QLatin1String(retValue.errorMessage()));
            QMessageBox::warning( this, tr("Warning while loading file"), text);
        }

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
    ui.groupPython->setEnabled(true);
    ui.scrollParamsMand->setEnabled(true);
    ui.scrollParamsOpt->setEnabled(true);
}

//------------------------------------------------------------------------------------------------------------
void DialogOpenFileWithFilter::closeEvent(QCloseEvent *e)
{
    if(filterCall.isRunning())
    {
        QMessageBox::critical(this, tr("Procedure still running"), tr("The file is still being loaded. Please wait..."));
        e->ignore();
    }
}

} //end namespace ito
