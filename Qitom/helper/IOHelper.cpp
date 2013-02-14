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

#include "../python/pythonEngineInc.h"
#include "../python/pythonEngine.h"
#include "../../PointCloud/pclStructures.h"

#include "../organizer/scriptEditorOrganizer.h"
#include "../organizer/processOrganizer.h"
#include "../organizer/addInManager.h"
#include "../organizer/algoInterfaceValidator.h"

#include "../ui/dialogOpenFileWithFilter.h"
#include "../ui/dialogSaveFileWithFilter.h"

#include "../AppManagement.h"
#include "IOHelper.h"

#include <qdir.h>
#include <qfile.h>
#include <qfiledialog.h>
#include <qfileinfo.h>
#include <qdesktopservices.h>
#include <qurl.h>
#include <qprocess.h>
#include <QtNetwork/qtcpsocket.h>
#include <QtNetwork/qhostaddress.h>

namespace ito {

/*static */RetVal IOHelper::openGeneralFile(QString generalFileName, bool openUnknownsWithExternalApp, bool showMessages, QWidget* parent, const char* errorSlotMemberOfParent, bool globalNotLocalWorkspace /*= true*/)
{
    QFile file(generalFileName);

    if (!file.exists())
    {
        if (showMessages)
        {
            QMessageBox msgBox(parent);
            msgBox.setText(tr("File '%1' does not exist").arg(generalFileName));
            msgBox.exec();
        }
        return RetVal(retError, 1001, tr("file does not exist").toAscii().data());
    }
    else
    {
        QFileInfo fileinfo(file);
        QString suffix = fileinfo.suffix();

        if (suffix == "py")
        {
            return openPythonScript(generalFileName);
        }
        else if (suffix == "idc") //itom data collection
        {
            return importPyWorkspaceVars(generalFileName, globalNotLocalWorkspace);
        }
        else if (suffix == "mat") //matlab file
        {
            return importPyWorkspaceVars(generalFileName, globalNotLocalWorkspace);
        }
        else if (suffix == "ui") //UI file
        {
            return openUIFile(generalFileName, parent, errorSlotMemberOfParent);
        }
        else //check whether there is a plugin which can open this file
        {
            ito::AddInManager *AIM = static_cast<ito::AddInManager*>(AppManagement::getAddinManager());
            ito::AddInAlgo::FilterDef *filter = NULL;
            if (AIM)
            {
                QList<ito::AddInAlgo::FilterDef*> filters = AIM->getFilterByInterface(ito::AddInAlgo::iReadDataObject, suffix);
                filters << AIM->getFilterByInterface(ito::AddInAlgo::iReadPointCloud, suffix);
                filters << AIM->getFilterByInterface(ito::AddInAlgo::iReadPolygonMesh, suffix);

                if (filters.size() == 1)
                {
                    filter = filters[0];
                }
                else if (filters.size() > 1)
                {
                    bool ok = false;
                    QStringList items;
                    foreach(const ito::AddInAlgo::FilterDef* f, filters)
                    {
                        items.append(f->m_name);
                    }
                    QString result = QInputDialog::getItem(parent, tr("Multiple plugins"), tr("Multiple plugins provide methods to load the file of type '%1'. Please choose one.").arg(suffix), items, 0, false, &ok);
                    if (ok)
                    {
                        filter = filters[items.indexOf(result)];
                    }
                    else
                    {
                        return ito::retOk;
                    }
                }

                if (filter)
                {
                    return uiOpenFileWithFilter(filter, generalFileName, parent);
                }
            }
        }

        if (openUnknownsWithExternalApp)
        {
            QUrl url = QUrl::fromLocalFile(generalFileName);
            if (QDesktopServices::openUrl(url))
            {
                return RetVal(retOk);
            }
            else
            {
                if (showMessages)
                {
                    QMessageBox msgBox(parent);
                    msgBox.setText(tr("File '%1' could not be opened with registered external application").arg(generalFileName));
                    msgBox.exec();
                }
                return RetVal(retError, 1002, tr("file could not be opened with external application").toAscii().data());
            }
        }
        else
        {
            if (showMessages)
            {
                QMessageBox msgBox(parent);
                msgBox.setText(tr("File '%1' can not be opened with this application").arg(generalFileName));
                msgBox.exec();
            }
            return RetVal(retError, 1002, tr("file can not be opened with this application").toAscii().data());
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static */RetVal IOHelper::uiExportPyWorkspaceVars(bool globalNotLocal, QStringList varNames, QVector<int> compatibleParamBaseTypes, QString defaultPath, QWidget* parent)
{
    if (defaultPath.isNull() || defaultPath.isEmpty()) defaultPath = QDir::currentPath();
    if (compatibleParamBaseTypes.size() == 0)
    {
        compatibleParamBaseTypes.fill(varNames.size(), 0);
    }

    Q_ASSERT(varNames.size() == compatibleParamBaseTypes.size());
    IOFilters filters = (IOFilters) (IOHelper::IOOutput | IOHelper::IOWorkspace | IOHelper::IOPlugin);

    if (varNames.size() == 1)
    {
        switch(compatibleParamBaseTypes[0])
        {
        case ito::ParamBase::DObjPtr:
            filters |= IOHelper::IOMimeDataObject;
            break;
        case ito::ParamBase::PointCloudPtr:
            filters |= IOHelper::IOMimePointCloud;
            break;
        case ito::ParamBase::PolygonMeshPtr:
            filters |= IOHelper::IOMimePolygonMesh;
            break;
        }
    }

    QString filterString = IOHelper::getFileFilters(filters);
    static QString selectedFilter; //this will save the recently selected filter for the next time
    QString filename = QFileDialog::getSaveFileName(parent, tr("Save selected variables as..."), defaultPath, filterString, &selectedFilter);

    if (filename.isEmpty())
    {
        return RetVal(retOk);
    }
    else
    {
        QFileInfo info(filename);
        if (info.suffix() == "idc" || info.suffix() == "mat")
        {
            QDir::setCurrent(info.path());
            return exportPyWorkspaceVars(filename, globalNotLocal, varNames);
        }
        else
        {
            //try to open it with filters
            RetVal retVal;
            QSharedPointer<ito::ParamBase> value;
            PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
            if (eng == NULL)
            {
                return RetVal(retError, 1, tr("python engine not available").toAscii().data());
            }

            if (eng->isPythonBusy() && !eng->isPythonDebuggingAndWaiting())
            {
                return RetVal(retError, 2, tr("variables cannot be imported since python is busy right now").toAscii().data());
            }

            QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
            //get values from workspace
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
            QSharedPointer<SharedParamBasePointerVector> values(new SharedParamBasePointerVector());
            QMetaObject::invokeMethod(eng, "getParamsFromWorkspace",Q_ARG(bool,globalNotLocal), Q_ARG(QStringList, varNames), Q_ARG(QVector<int>, compatibleParamBaseTypes), Q_ARG(QSharedPointer<SharedParamBasePointerVector>, values), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
            if (!locker.getSemaphore()->wait(5000))
            {
                retVal += RetVal(retError, 0, tr("timeout while getting value from workspace").toAscii().data());
            }
            else
            {
                retVal += locker.getSemaphore()->returnValue;
            }
            
            if (values->size() != varNames.size())
            {
                retVal += RetVal(retError, 0, tr("the number of values returned from workspace does not correspond to requested number").toAscii().data());
            }
            QApplication::restoreOverrideCursor();

            for (int i=0;i<varNames.size() && !retVal.containsError();i++)
            {
                retVal += uiSaveFileWithFilter((*values)[0], filename,parent);
            }

            if (retVal.containsError())
            {
                QString text = tr("An error occurred while saving to file.");
                if (retVal.errorMessage()) text.append("\n").append(retVal.errorMessage());
                QMessageBox::critical(parent, tr("Error while saving file"), text);
            }

            return retVal;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static */RetVal IOHelper::exportPyWorkspaceVars(QString filename, bool globalNotLocal, QStringList varNames)
{
    RetVal retValue(retOk);

    QFile file(filename);

    if (!file.open(QIODevice::WriteOnly))
    {
        return RetVal(retError, 3, tr("file cannot be opened").toAscii().data());
    }
    file.close();

    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if (eng == NULL)
    {
        return RetVal(retError, 1, tr("python engine not available").toAscii().data());
    }

    if (eng->isPythonBusy() && !eng->isPythonDebuggingAndWaiting())
    {
        return RetVal(retError, 2, tr("variables cannot be exported since python is busy right now").toAscii().data());
    }

    QFileInfo info(file);

    if (info.suffix() == "idc")
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore(1));

        QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
        QMetaObject::invokeMethod(eng, "pickleVariables", Q_ARG(bool,globalNotLocal), Q_ARG(QString,filename), Q_ARG(QStringList,varNames), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        if (!locker.getSemaphore()->wait(120000))
        {
            retValue += RetVal(retError, 2, tr("timeout while pickling variables").toAscii().data());
        }

        QApplication::restoreOverrideCursor();
    }
    else if (info.suffix() == "mat")
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore(1));

        QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
        QMetaObject::invokeMethod(eng, "saveMatlabVariables", Q_ARG(bool,globalNotLocal), Q_ARG(QString,filename), Q_ARG(QStringList,varNames), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));

        if (!locker.getSemaphore()->wait(120000))
        {
            retValue += RetVal(retError, 2, tr("timeout while saving variables to matlab file").toAscii().data());
        }

        QApplication::restoreOverrideCursor();
    }
    else
    {
        retValue += RetVal(retError, 0, tr("suffix must be *.idc or *.mat").toAscii().data());
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static */RetVal IOHelper::uiImportPyWorkspaceVars(bool globalNotLocal, IOFilters IOfilters, QString defaultPath, QWidget* parent)
{
    if (defaultPath.isNull() || defaultPath.isEmpty()) defaultPath = QDir::currentPath();

    IOfilters ^= ito::IOHelper::IOOutput;
    QString filters = IOHelper::getFileFilters(IOfilters);
    static QString selectedFilter; //this variable will contain the last selected filter which is the default for the next time

    QString filename = QFileDialog::getOpenFileName(parent, tr("Import data"), defaultPath, filters, &selectedFilter);

    if (filename.isEmpty())
    {
        return RetVal(retOk);
    }
    else
    {
        QFile file(filename);
        if (file.exists())
        {
            QDir::setCurrent(QFileInfo(filename).path());
            QFileInfo info(filename);
            return openGeneralFile(filename, false, true, parent, NULL, globalNotLocal);
        }
        else
        {
            return RetVal(retError, 1, tr("file not found").toAscii().data());
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static */RetVal IOHelper::importPyWorkspaceVars(QString filename, bool globalNotLocal)
{
    RetVal retValue(retOk);

    QFile file(filename);

    if (!file.open(QIODevice::ReadOnly))
    {
        return RetVal(retError, 3, tr("file cannot be opened").toAscii().data());
    }
    file.close();

    PythonEngine* eng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
    if (eng == NULL)
    {
        return RetVal(retError, 1, tr("python engine not available").toAscii().data());
    }

    if (eng->isPythonBusy() && !eng->isPythonDebuggingAndWaiting())
    {
        return RetVal(retError, 2, tr("variables cannot be imported since python is busy right now").toAscii().data());
    }

    QFileInfo info(file);

    if (info.suffix() == "idc")
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore(1));

        QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
        QMetaObject::invokeMethod(eng, "unpickleVariables", Q_ARG(bool,globalNotLocal), Q_ARG(QString,filename), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
        if (!locker.getSemaphore()->wait(120000))
        {
            retValue += RetVal(retError, 2, tr("timeout while unpickling variables").toAscii().data());
        }

        QApplication::restoreOverrideCursor();
    }
    else if (info.suffix() == "mat")
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore(1));

        QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
        QMetaObject::invokeMethod(eng, "loadMatlabVariables", Q_ARG(bool,globalNotLocal), Q_ARG(QString,filename), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore()));
        if (!locker.getSemaphore()->wait(120000))
        {
            retValue += RetVal(retError, 2, tr("timeout while loading matlab variables").toAscii().data());
        }

        QApplication::restoreOverrideCursor();
    }
    else
    {
        retValue += RetVal(retError, 0, tr("suffix must be *.idc or *.mat").toAscii().data());
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static */RetVal IOHelper::uiOpenPythonScript(QString defaultPath, QWidget* parent)
{
    QString fileName;

    if (defaultPath.isNull() || defaultPath.isEmpty()) defaultPath = QDir::currentPath();

    fileName = QFileDialog::getOpenFileName(parent, tr("open python script"), defaultPath, tr("python (*.py)"));

    QFileInfo info(fileName);

    if (fileName.isEmpty())
    {
        return RetVal(retOk);
    }
    else
    {
        QDir::setCurrent(QFileInfo(fileName).path());
        return openPythonScript(fileName);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static */RetVal IOHelper::openPythonScript(QString filename)
{
    QFile file(filename);
    RetVal retValue(retOk);

    if (file.exists())
    {
        QObject *seo = AppManagement::getScriptEditorOrganizer();

        if (seo != NULL)
        {
            ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore(1));

            QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
            QMetaObject::invokeMethod(seo,"openScript",Q_ARG(QString, filename),Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
            if (!locker.getSemaphore()->wait(20000))
            {
                retValue += RetVal(retError, 2, tr("timeout while opening script").toAscii().data());
            }
            QApplication::restoreOverrideCursor();

            return retValue;
        }

        return RetVal(retError);
    }
    else
    {
        return RetVal(retError);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static */RetVal IOHelper::openUIFile(QString filename, QWidget* parent, const char* errorSlotMemberOfParent)
{
    ProcessOrganizer *po = qobject_cast<ProcessOrganizer*>(AppManagement::getProcessOrganizer());
    if (po)
    {
        bool existingProcess = false;
        QProcess *process = po->getProcess("designer", true, existingProcess, false);

        if (existingProcess && process->state() == QProcess::Running)
        {
            //the output message is mainly useful in order to get the socket-number of the designer (started as server)
            //the designer returns this number over stdOut. When opening another ui-file, this ui-file is send over a socket
            //connection
            QString tcpPort = po->getStandardOutputBuffer("designer");
            bool done = false;
            if (tcpPort != "")
            {
                QTcpSocket socket;
                quint16 tcpPort2 = tcpPort.toInt();
                if (tcpPort2 != 0)
                {
                    QString ipAddress = QHostAddress(QHostAddress::LocalHost).toString();
                    socket.connectToHost(ipAddress, tcpPort2, QIODevice::WriteOnly);
                    bool result = socket.waitForConnected(30000);
                    if (result)
                    {
                        filename.append("\n");
                        socket.write(filename.toAscii().data());
                        done = true;
                        socket.disconnectFromHost();
                        socket.waitForDisconnected(250);

                        //Q_PID processID = process->pid();
                        po->bringWindowsOnTop("Qt Designer");
                    }
                }
            }

            if (!done)
            {
                process = po->getProcess("designer", false, existingProcess, false);
                //create new process for designer, since sending data to existing one failed
                QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
                QString appPath = QDir::cleanPath(QCoreApplication::applicationDirPath());
                env.insert("QT_PLUGIN_PATH", appPath);
                process->setProcessEnvironment(env);

                if (errorSlotMemberOfParent != NULL)
                {
                    connect(process, SIGNAL(error(QProcess::ProcessError)), parent, errorSlotMemberOfParent);
                }

                po->clearStandardOutputBuffer("designer");

                QStringList arguments;
                arguments << "-server" << filename;
                process->start(QLatin1String("designer"), arguments);
            }
        }
        else
        {
            QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
            QString appPath = QDir::cleanPath(QCoreApplication::applicationDirPath());
            env.insert("QT_PLUGIN_PATH", appPath);
            process->setProcessEnvironment(env);

            if (errorSlotMemberOfParent != NULL)
            {
                connect(process, SIGNAL(error(QProcess::ProcessError)), parent, errorSlotMemberOfParent);
            }

            po->clearStandardOutputBuffer("designer");

            QStringList arguments;
            arguments << "-server" << filename;
            process->start(QLatin1String("designer"), arguments);
        }
    }

    return RetVal(retOk);
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ RetVal IOHelper::uiOpenFileWithFilter(ito::AddInAlgo::FilterDef *filter, const QString &filename, QWidget *parent /*= NULL*/)
{
    RetVal retval;
    ito::AddInManager *AIM = static_cast<ito::AddInManager*>(AppManagement::getAddinManager());
    PythonEngine *pyEng = static_cast<PythonEngine*>(AppManagement::getPythonEngine());
    QVector<ito::ParamBase> autoMand;
    QVector<ito::ParamBase> autoOut;
    QVector<ito::Param> userMand;
    QVector<ito::Param> userOpt;
    DialogOpenFileWithFilter *dialog = NULL;
    QStringList pythonVarNames;
    bool putParamsToWorkspace = false;
    QVector<SharedParamBasePointer> values;

    ito::DataObject dObj;
    ito::PCLPointCloud pointCloud;
    ito::PCLPolygonMesh polygonMesh;

    if (AIM && pyEng)
    {
        const ito::FilterParams *fp = AIM->getHashedFilterParams(filter->m_paramFunc);
        autoOut.clear();
        autoMand.clear();
        const AlgoInterfaceValidator *aiv = AIM->getAlgoInterfaceValidator();
        if (aiv)
        {
            retval += aiv->getInterfaceParameters(filter->m_interface, autoMand, autoOut);
            if (!retval.containsError())
            {
                userOpt = fp->paramsOpt;
                userMand = fp->paramsMand.mid(autoMand.size());
            }

            switch (filter->m_interface)
            {
            case ito::AddInAlgo::iReadDataObject:
                {
                    //1. dataObject
                    autoMand[0].setVal<void*>(&dObj);
                    //2. filename
                    autoMand[1].setVal<char*>(filename.toAscii().data());

                    dialog = new DialogOpenFileWithFilter(filename, filter, autoMand, autoOut, userMand, userOpt, retval, parent );
                    if (!retval.containsError())
                    {
                        int result = dialog->exec();
                        if (result == QDialog::Accepted)
                        {
                            //autoMand = dialog->getAutoMand();
                            //autoOut = dialog->getAutoOut();
                            pythonVarNames << dialog->getPythonVariable();
                            values << SharedParamBasePointer(new ito::ParamBase(autoMand[0]));
                            putParamsToWorkspace = true;
                        }
                    }
                }
                break;
            case ito::AddInAlgo::iReadPointCloud:
                {
                    //1. pointCloud
                    autoMand[0].setVal<void*>(&pointCloud);
                    //2. filename
                    autoMand[1].setVal<char*>(filename.toAscii().data());

                    dialog = new DialogOpenFileWithFilter(filename, filter, autoMand, autoOut, userMand, userOpt, retval, parent );
                    if (!retval.containsError())
                    {
                        int result = dialog->exec();
                        if (result == QDialog::Accepted)
                        {
                            //autoMand = dialog->getAutoMand();
                            //autoOut = dialog->getAutoOut();
                            pythonVarNames << dialog->getPythonVariable();
                            values << SharedParamBasePointer(new ito::ParamBase(autoMand[0]));
                            putParamsToWorkspace = true;
                        }
                    }
                }
                break;
            case ito::AddInAlgo::iReadPolygonMesh:
                {
                    //1. polygonMesh
                    autoMand[0].setVal<void*>(&polygonMesh);
                    //2. filename
                    autoMand[1].setVal<char*>(filename.toAscii().data());

                    dialog = new DialogOpenFileWithFilter(filename, filter, autoMand, autoOut, userMand, userOpt, retval, parent);
                    if (!retval.containsError())
                    {
                        int result = dialog->exec();
                        if (result == QDialog::Accepted)
                        {
                            //autoMand = dialog->getAutoMand();
                            //autoOut = dialog->getAutoOut();
                            pythonVarNames << dialog->getPythonVariable();
                            values << SharedParamBasePointer(new ito::ParamBase(autoMand[0]));
                            putParamsToWorkspace = true;
                        }
                    }
                }
                break;
            default:
                retval += ito::RetVal(ito::retError, 0, tr("The algorithm interface is not supported").toAscii().data());
            }

            if (!retval.containsError() && putParamsToWorkspace)
            {
                ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
                        
                QMetaObject::invokeMethod(pyEng, "putParamsToWorkspace", Q_ARG(bool,true), Q_ARG(QStringList, pythonVarNames), Q_ARG(QVector<SharedParamBasePointer>, values), Q_ARG(ItomSharedSemaphore*,locker.getSemaphore()));
                if (locker.getSemaphore()->wait(10000) == false)
                {
                    QMessageBox::critical(parent, tr("Timeout while sending values to python"), tr("A timeout occurred while content of loaded file has been sent to python workspace"));
                }
                else
                {
                    retval += locker.getSemaphore()->returnValue;
                    if (retval != ito::retOk)
                    {
                        QString title;
                        QString text;
                        if (retval.errorMessage()) text = QString("\n%1").arg(retval.errorMessage());
                        if (retval.containsError())
                        {
                            text.prepend(tr("An error occured while sending the content of the loaded file to the python workspace."));
                            QMessageBox::critical(parent, tr("Error while sending values to python"), text);
                        }
                        else if (retval.containsWarning())
                        {
                            text.prepend(tr("A warning occured while sending the content of the loaded file to the python workspace."));
                            QMessageBox::warning(parent, tr("Warning while sending values to python"), text);
                        }
                    }
                }
            }

            DELETE_AND_SET_NULL(dialog);
        }
        else
        {
            retval += RetVal(retError, 0, tr("AlgoInterfaceValidator not available.").toAscii().data());
        }

        if (!retval.containsError())
        {
            
        }
    }
    else
    {
        retval += RetVal(retError, 0, tr("AddInManager or PythonEngine not available").toAscii().data());
    }
    
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ RetVal IOHelper::uiSaveFileWithFilter(QSharedPointer<ito::ParamBase> &value, const QString &filename, QWidget *parent /*= NULL*/)
{

    ito::AddInManager *AIM = static_cast<ito::AddInManager*>(AppManagement::getAddinManager());
    ito::AddInAlgo::FilterDef *filter = NULL;
    QList<ito::AddInAlgo::FilterDef*> filters;
    QFileInfo info(filename);
    QString suffix = info.suffix();

    if (AIM)
    {
        switch(value->getType())
        {
        case ito::ParamBase::DObjPtr & ito::paramTypeMask:
            filters = AIM->getFilterByInterface(ito::AddInAlgo::iWriteDataObject, suffix);
            break;
        case ito::ParamBase::PointCloudPtr & ito::paramTypeMask:
            filters = AIM->getFilterByInterface(ito::AddInAlgo::iWritePointCloud, suffix);
            break;
        case ito::ParamBase::PolygonMeshPtr & ito::paramTypeMask:
            filters = AIM->getFilterByInterface(ito::AddInAlgo::iWritePolygonMesh, suffix);
            break;
        default:
            return RetVal(retError, 0, tr("There is no plugin interface able to save the requested file type").toAscii().data());
        }

        if (filters.size() == 1)
        {
            filter = filters[0];
        }
        else if (filters.size() > 1)
        {
            bool ok = false;
            QStringList items;
            foreach(const ito::AddInAlgo::FilterDef* f, filters)
            {
                items.append(f->m_name);
            }
            QString result = QInputDialog::getItem(parent, tr("Multiple plugins"), tr("Multiple plugins provide methods to save the file of type '%1'. Please choose one.").arg(suffix), items, 0, false, &ok);
            if (ok)
            {
                filter = filters[items.indexOf(result)];
            }
            else
            {
                return ito::retOk; //user cancel
            }
        }

        QVector<ito::ParamBase> autoMand;
        QVector<ito::ParamBase> autoOut;
        QVector<ito::ParamBase> paramsOpt;
        QVector<ito::ParamBase> paramsMand;
        QVector<ito::Param> userMand;
        QVector<ito::Param> userOpt;
        ito::RetVal retval;

        const ito::FilterParams *fp = AIM->getHashedFilterParams(filter->m_paramFunc);
        const AlgoInterfaceValidator *aiv = AIM->getAlgoInterfaceValidator();

        if (aiv)
        {
            retval += aiv->getInterfaceParameters(filter->m_interface, autoMand, autoOut);
            if (!retval.containsError())
            {
                switch(filter->m_interface)
                {
                case ito::AddInAlgo::iWriteDataObject:
                case ito::AddInAlgo::iWritePolygonMesh:
                case ito::AddInAlgo::iWritePointCloud:
                    autoMand[0] = *value;
                    autoMand[1].setVal<char*>(filename.toAscii().data());
                    break;
                default:
                    retval += ito::RetVal(retError, 0, tr("algorithm interface not supported").toAscii().data());
                }

                if (!retval.containsError())
                {
                    userOpt = fp->paramsOpt;
                    userMand = fp->paramsMand.mid(autoMand.size());

                    if (userMand.size() > 0 || userOpt.size() > 0)
                    {
                        DialogSaveFileWithFilter *dialog = new DialogSaveFileWithFilter(filename, filter, autoMand, autoOut, userMand, userOpt, parent);
                        if (dialog->exec() == QDialog::Accepted)
                        {
                            QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
                            dialog->getParameters(paramsMand, paramsOpt);
                            paramsMand = autoMand + paramsMand;
                            retval += filter->m_filterFunc(&paramsMand, &paramsOpt, &autoOut);
                            QApplication::restoreOverrideCursor();
                        }

                        DELETE_AND_SET_NULL(dialog);
                    }
                    else
                    {
                        QApplication::setOverrideCursor(QCursor(Qt::WaitCursor));
                        retval += filter->m_filterFunc(&autoMand, &paramsOpt, &autoOut);
                        QApplication::restoreOverrideCursor();
                    }
                }
            }
            else
            {
                retval += RetVal(retError, 0, tr("error while getting mand and out parameters from algorithm interface").toAscii().data());
            }
        }
        else
        {
            retval += RetVal(retError, 0, tr("AlgoInterfaceValidator not available").toAscii().data());
        }
        return retval;
    }
    else
    {
        return RetVal(retError, 0, tr("AddInManager not available").toAscii().data());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ QString IOHelper::getFileFilters(IOFilters IOfilters)
{
    QStringList filter;

    if (IOfilters.testFlag(ito::IOHelper::IOWorkspace) == false)
    {
        filter << tr("Python Scripts (*.py)") << tr("Itom Data Collection (*.idc)") << tr("Matlab Matrix (*.mat)") << tr("User Interfaces (*.ui)");
    }
    else
    {
        filter << tr("Itom Data Collection (*.idc)") << tr("Matlab Matrix (*.mat)");
    }

    if (IOfilters.testFlag(ito::IOHelper::IOPlugin))
    {
        ito::AddInManager *AIM = static_cast<ito::AddInManager*>(AppManagement::getAddinManager());
        QList<ito::AddInAlgo::FilterDef*> filterPlugins;
        if (AIM)
        {
            if (IOfilters.testFlag(ito::IOHelper::IOInput))
            {
                if (IOfilters.testFlag(ito::IOHelper::IOMimeDataObject))
                {
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iReadDataObject, QString::Null());
                    foreach(const ito::AddInAlgo::FilterDef *d, filterPlugins)
                    {
                        filter << d->m_interfaceMeta.split(";;"); //in case of iReadDataObject-interface, m_interfaceMeta contains the file-filter
                    }
                }
                if (IOfilters.testFlag(ito::IOHelper::IOMimePointCloud))
                {
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iReadPointCloud, QString::Null());
                    foreach(const ito::AddInAlgo::FilterDef *d, filterPlugins)
                    {
                        filter << d->m_interfaceMeta.split(";;"); //in case of iReadDataObject-interface, m_interfaceMeta contains the file-filter
                    }
                }
                if (IOfilters.testFlag(ito::IOHelper::IOMimePolygonMesh))
                {
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iReadPolygonMesh, QString::Null());
                    foreach(const ito::AddInAlgo::FilterDef *d, filterPlugins)
                    {
                        filter << d->m_interfaceMeta.split(";;"); //in case of iReadDataObject-interface, m_interfaceMeta contains the file-filter
                    }
                }
            }

            if (IOfilters.testFlag(ito::IOHelper::IOOutput))
            {
                if (IOfilters.testFlag(ito::IOHelper::IOMimeDataObject))
                {
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iWriteDataObject, QString::Null());
                    foreach(const ito::AddInAlgo::FilterDef *d, filterPlugins)
                    {
                        filter << d->m_interfaceMeta.split(";;"); //in case of iWriteDataObject-interface, m_interfaceMeta contains the file-filter
                    }
                }
                if (IOfilters.testFlag(ito::IOHelper::IOMimePointCloud))
                {
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iWritePointCloud, QString::Null());
                    foreach(const ito::AddInAlgo::FilterDef *d, filterPlugins)
                    {
                        filter << d->m_interfaceMeta.split(";;"); //in case of iReadDataObject-interface, m_interfaceMeta contains the file-filter
                    }
                }
                if (IOfilters.testFlag(ito::IOHelper::IOMimePolygonMesh))
                {
                    filterPlugins = AIM->getFilterByInterface(ito::AddInAlgo::iWritePolygonMesh, QString::Null());
                    foreach(const ito::AddInAlgo::FilterDef *d, filterPlugins)
                    {
                        filter << d->m_interfaceMeta.split(";;"); //in case of iReadDataObject-interface, m_interfaceMeta contains the file-filter
                    }
                }
            }

            //add here further filters
        }
    }
    
    if (IOfilters.testFlag(ito::IOHelper::IOWorkspace) == false)
    {
        filter << tr("Itom Files (*.py *.idc *.mat *.ui)");
    }
    else
    {
        filter << tr("Itom Files (*.idc *.mat)");
    }

    if (IOfilters.testFlag(ito::IOHelper::IOAllFiles))
    {
        filter << tr("All Files (*.*)");
    }

    //delete duplicates
    filter.removeDuplicates();  

    return filter.join(";;");
}

} //end namespace ito
