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

//import this before any qobject stuff
#include "../python/pythonEngine.h"
#include "../python/pythonQtConversion.h"

#include "pipManager.h"

#include "../../common/sharedStructures.h"
#include "../AppManagement.h"
#include <qdir.h>
#include <qsettings.h>
#include <QProcessEnvironment>

namespace ito
{

    //----------------------------------------------------------------------------------------------------------------------------------
    /** constructor
    *
    *   contructor, creating column headers for the tree view
    */
    PipManager::PipManager(ito::RetVal &retval, QObject *parent /*= 0*/) :
        QAbstractItemModel(parent),
        m_currentTask(taskNo),
        m_pipAvailable(false),
        m_pipVersion(0x000000),
        m_pUserDefinedPythonHome(NULL)
    {
        m_headers << tr("Name") << tr("Version") << tr("Location") << tr("Requires") << tr("Updates") << tr("Summary") << tr("Homepage") << tr("License");
        m_alignment << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft);

        connect(&m_pipProcess, SIGNAL(error(QProcess::ProcessError)), this, SLOT(processError(QProcess::ProcessError)));
        connect(&m_pipProcess, SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(processFinished(int, QProcess::ExitStatus)));
        connect(&m_pipProcess, SIGNAL(readyReadStandardError()), this, SLOT(processReadyReadStandardError()));
        connect(&m_pipProcess, SIGNAL(readyReadStandardOutput()), this, SLOT(processReadyReadStandardOutput()));

        const PythonEngine *pyeng = qobject_cast<PythonEngine*>(AppManagement::getPythonEngine());
        if (pyeng)
        {
            m_pythonPath = pyeng->getPythonExecutable();
        }
        else
        {
            retval += initPythonIfStandalone();

            if (!retval.containsError())
            {
                //Pip Manager has been started as standalone application to update packages like Numpy that cannot be updated if itom is running and the Python Engine has been entirely started.
                Py_Initialize();
                if (Py_IsInitialized())
                {
#if defined WIN32
                    //on windows, sys.executable returns the path of qitom.exe. The absolute path to python.exe is given by sys.exec_prefix
                    PyObject *python_path_prefix = PySys_GetObject("exec_prefix"); //borrowed reference
                    if (python_path_prefix)
                    {
                        bool ok;
                        m_pythonPath = PythonQtConversion::PyObjGetString(python_path_prefix, true, ok);
                        if (ok)
                        {
                            QDir pythonPath(m_pythonPath);
                            if (pythonPath.exists())
                            {
                                m_pythonPath = pythonPath.absoluteFilePath("python.exe");
                            }
                            else
                            {
                                m_pythonPath = QString();
                            }
                        }
                        else
                        {
                            m_pythonPath = QString();
                        }
                    }
#elif defined linux
                    //on linux, sys.executable returns the absolute path to the python application, even in an embedded mode.
                    PyObject *python_executable = PySys_GetObject("executable"); //borrowed reference
                    if (python_executable)
                    {
                        bool ok;
                        m_pythonPath = PythonQtConversion::PyObjGetString(python_executable, true, ok);
                        if (!ok)
                        {
                            m_pythonPath = QString();
                        }
                    }
#else //APPLE
                    //on apple, sys.executable returns the absolute path to the python application, even in an embedded mode. (TODO: Check this assumption)
                    PyObject *python_executable = PySys_GetObject("executable"); //borrowed reference
                    if (python_executable)
                    {
                        bool ok;
                        m_pythonPath = PythonQtConversion::PyObjGetString(python_executable, true, ok);
                        if (!ok)
                        {
                            m_pythonPath = QString();
                        }
                    }
#endif
                    Py_Finalize();
                }
            }
        }

        if (!retval.containsError())
        {
            QString pythonHome = QString::fromWCharArray(Py_GetPythonHome());
            if (pythonHome != "")
            {
                QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
                env.insert("PYTHONHOME", pythonHome); // Add an environment variable
                m_pipProcess.setProcessEnvironment(env);
            }
            
        }
    }

//----------------------------------------------------------------------------------------------------------------------------------
/** destructor - clean up, clear header and alignment list
*
*/
PipManager::~PipManager()
{
    if (m_pipProcess.state() == QProcess::Running || m_pipProcess.state() == QProcess::Starting)
    {
        m_pipProcess.kill();
        m_pipProcess.waitForFinished(2000);
    }
}


//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal PipManager::initPythonIfStandalone()
{
    ito::RetVal retval;

	//keep this method consistent to PythonEngine::pythonSetup

	QString pythonSubDir = QCoreApplication::applicationDirPath() + QString("/python%1").arg(PY_MAJOR_VERSION);
	QString pythonAllInOneDir = QCoreApplication::applicationDirPath() + QString("/../../3rdParty/Python");
	qDebug() << "pythonAllInOneDir:" << pythonAllInOneDir;
	//check if an alternative home directory of Python should be set:
	QSettings settings(AppManagement::getSettingsFile(), QSettings::IniFormat);
	settings.beginGroup("Python");
	QString pythonHomeFromSettings = settings.value("pyHome", "").toString();
	int pythonDirState = settings.value("pyDirState", -1).toInt();
	if (pythonDirState == -1) //not yet decided
	{
#ifdef WIN32
		if (QDir(pythonSubDir).exists() && \
			QFileInfo(pythonSubDir + QString("/python%1%2.dll").arg(PY_MAJOR_VERSION).arg(PY_MINOR_VERSION)).exists())
		{
			pythonDirState = 0; //use pythonXX subdirectory of itom as python home path
	}
		else if (QDir(pythonAllInOneDir).exists() && \
			QFileInfo(pythonAllInOneDir + QString("/python%1%2.dll").arg(PY_MAJOR_VERSION).arg(PY_MINOR_VERSION)).exists())
		{
			pythonDirState = 2;
			pythonHomeFromSettings = pythonAllInOneDir;
			settings.setValue("pyHome", pythonHomeFromSettings);
		}
		else
		{
			pythonDirState = 1; //use python default search mechanism for home path (e.g. registry...)
		}
#else
		pythonDirState = 1;
#endif
		qDebug() << "pythonDirState:" << pythonDirState;
		qDebug() << "pythonHomeFromSettings:" << pythonHomeFromSettings;
		settings.setValue("pyDirState", pythonDirState);
}

	settings.endGroup();

	QString pythonDir = "";
	if (pythonDirState == 0) //use pythonXX subdirectory of itom as python home path
	{
		if (QDir(pythonSubDir).exists())
		{
			pythonDir = pythonSubDir;
		}
		else
		{
			retval += RetVal::format(retError, 0, tr("The itom subdirectory of Python '%s' is not existing.\nPlease change setting in the property dialog of itom.").toLatin1().data(),
				pythonSubDir.toLatin1().data());
			return retval;
		}
	}
	else if (pythonDirState == 2) //user-defined value
	{

		if (QDir(pythonHomeFromSettings).exists())
		{
			pythonDir = pythonHomeFromSettings;
		}
		else
		{
			retval += RetVal::format(retError, 0, tr("Settings value Python::pyHome has not been set as Python Home directory since it does not exist:  %s").toLatin1().data(),
				pythonHomeFromSettings.toLatin1().data());
			return retval;
		}
	}

	if (pythonDir != "")
	{
		//the python home path given to Py_SetPythonHome must be persistent for the whole Python session
#if PY_VERSION_HEX < 0x03050000
		m_pUserDefinedPythonHome = (wchar_t*)PyMem_RawMalloc((pythonDir.size() + 10) * sizeof(wchar_t));
		memset(m_pUserDefinedPythonHome, 0, (pythonDir.size() + 10) * sizeof(wchar_t));
		pythonDir.toWCharArray(m_pUserDefinedPythonHome);
#else
		m_pUserDefinedPythonHome = Py_DecodeLocale(pythonDir.toLatin1().data(), NULL);
#endif
		Py_SetPythonHome(m_pUserDefinedPythonHome);
	}

	//read directory values from Python
	qDebug() << "Py_GetPythonHome:" << QString::fromWCharArray(Py_GetPythonHome());
	qDebug() << "Py_GetPath:" << QString::fromWCharArray(Py_GetPath());
	qDebug() << "Py_GetProgramName:" << QString::fromWCharArray(Py_GetProgramName());

	//check PythonHome to prevent crash upon initialization of Python:
	QString pythonHome = QString::fromWCharArray(Py_GetPythonHome());
#ifdef WIN32
	QStringList pythonPath = QString::fromWCharArray(Py_GetPath()).split(";");
#else
	QStringList pythonPath = QString::fromWCharArray(Py_GetPath()).split(":");
#endif
	QDir pythonHomeDir(pythonHome);
	bool pythonPathValid = false;
	if (!pythonHomeDir.exists() && pythonHome != "")
	{
		retval += RetVal::format(retError, 0, tr("The home directory of Python is currently set to the non-existing directory '%s'\nPython cannot be started. Please set either the environment variable PYTHONHOME to the base directory of python \nor correct the base directory in the property dialog of itom.").toLatin1().data(),
			pythonHomeDir.absolutePath().toLatin1().data());
		return retval;
	}

	foreach(const QString &path, pythonPath)
	{
		QDir pathDir(path);
		if (pathDir.exists("os.py") || pathDir.exists("os.pyc"))
		{
			pythonPathValid = true;
			break;
		}
	}

	if (!pythonPathValid)
	{
		retval += RetVal::format(retError, 0, tr("The built-in library path of Python could not be found. The current home directory is '%s'\nPython cannot be started. Please set either the environment variable PYTHONHOME to the base directory of python \nor correct the base directory in the preferences dialog of itom.").toLatin1().data(),
			pythonHomeDir.absolutePath().toLatin1().data());
		return retval;
	}

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return parent element
*   @param [in] index   the element's index for which the parent should be returned
*   @return     the parent element. 
*
*/
QModelIndex PipManager::parent(const QModelIndex &index) const
{
    return QModelIndex();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return number of rows
*   @param [in] parent parent of current item
*   @return     returns number of users
*/
int PipManager::rowCount(const QModelIndex &parent) const
{
    return m_pythonPackages.length();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return the header / captions for the tree view model
*
*/
QVariant PipManager::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role == Qt::DisplayRole && orientation == Qt::Horizontal )
    {
        if (section >= 0 && section < m_headers.size())
        {
            return m_headers.at(section);
        }
        return QVariant();
    }
    return QVariant();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return data elements for a given row
*   @param [in] index   index for which the data elements should be delivered
*   @param [in] role    the current role of the model 
*   @return data of the selected element, depending on the element's row and column (passed in index.row and index.column)
*
*/
QVariant PipManager::data(const QModelIndex &index, int role) const
{
    if(!index.isValid())
    {
        return QVariant();
    }
 
    if(role == Qt::DisplayRole || role == Qt::ToolTipRole)
    {
        const PythonPackage &package = m_pythonPackages[index.row()];
        switch (index.column())
        {
            case 0:
                return package.m_name;
            case 1:
                return package.m_version;
            case 2:
                return package.m_location;
            case 3:
                return package.m_requires;
            case 4:
                {
                    if (package.m_status == PythonPackage::Uptodate)
                    {
                        return tr("up to date");
                    }
                    else if (package.m_status == PythonPackage::Outdated)
                    {
                        return tr("new version %1 available").arg(package.m_newVersion);
                    }
                    else
                    {
                        return tr("unknown");
                    }
                }
            case 5:
                return package.m_summary;
            case 6:
                return package.m_homepage;
            case 7:
                return package.m_license;
            default:
                return QVariant();
        }
    }
    else if (role == Qt::UserRole + 1)
    {
        const PythonPackage &package = m_pythonPackages[index.row()];
        return (package.m_status == PythonPackage::Outdated);
    }
    
    return QVariant();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return column count
*   @param [in] parent parent of current item
*   @return     2 for child elements (instances) and the header size for root elements (plugins)
*/
int PipManager::columnCount(const QModelIndex & /*parent*/) const
{
    return m_headers.size();
}

//----------------------------------------------------------------------------------------------------------------------------------
/** return current index element
*   @param [in] row row of current element
*   @param [in] column column of current element
*   @param [in] parent  parent of current element
*   @return QModelIndex - element at current index
*
*   returns the passed row as index, as the users are arranged in a simple one dimensional list
*/
QModelIndex PipManager::index(int row, int column, const QModelIndex &parent) const
{
    if(parent.isValid() || row < 0 || row >= m_pythonPackages.length() || column < 0 || column >= m_headers.size())
    {
        return QModelIndex();
    }
    
    return createIndex(row, column);
}

//----------------------------------------------------------------------------------------------------------------------------------
bool PipManager::isPipStarted() const
{
    return m_pipProcess.pid() != 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::checkPipAvailable(const PipGeneralOptions &options /*= PipGeneralOptions()*/)
{
    if (m_pythonPath == "")
    {
        emit outputAvailable("Python is not available\n", false);
        return;
    }

    if (m_currentTask == taskNo)
    {
#if WIN32
        if (PY_VERSION_HEX >= 0x03030000 && PY_VERSION_HEX <= 0x03049999)
        {
            emit pipRequestStarted(taskCheckAvailable, "For Python 3.3 and 3.4, some packages (e.g. Scipy or OpenCV) might depend on the Microsoft Visual C++ 2010 redistributable package. Please install it if not yet done.\n\nCheck connection to pip and get version...\n");
        }
        else if (PY_VERSION_HEX >= 0x03050000)
        {
            emit pipRequestStarted(taskCheckAvailable, "For Python 3.5 or higher, some packages (e.g. Scipy or OpenCV) might depend on the Microsoft Visual C++ 2015 redistributable package. Please install it if not yet done.\n\nCheck connection to pip and get version...\n");
        }
        else
        {
            emit pipRequestStarted(taskCheckAvailable, "Check connection to pip and get version...\n");
        }
#else
        emit pipRequestStarted(taskCheckAvailable, "Check connection to pip and get version...\n");
#endif
        
        clearBuffers();
        m_currentTask = taskCheckAvailable;

        QStringList arguments;
        arguments << "-m" << "pip" << "-V";
        arguments << parseGeneralOptions(options, true);
        m_pipProcess.start(m_pythonPath, arguments);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::listAvailablePackages(const PipGeneralOptions &options /*= PipGeneralOptions()*/)
{
    if (m_pythonPath == "")
    {
        emit outputAvailable("Python is not available\n", false);
        return;
    }

    if (m_pipAvailable == false)
    {
        emit outputAvailable("pip is not available\n", false);
        return;
    }


    //list consists of two steps:
    //1. get package names using list
    //2. get more information using show package1 package2 ...
    if (m_currentTask == taskNo)
    {
        emit pipRequestStarted(taskListPackages1, "Get list of installed packages... (step 1)\n", true);
        clearBuffers();
        m_currentTask = taskListPackages1;
        m_generalOptionsCache = options;

        QStringList arguments;
        arguments << "-m" << "pip" << "list"; //here the pip version check is done

        if (m_pipVersion >= 0x120000) // >= 18.0
        {
            arguments << "--format=columns";
        }
		else if (m_pipVersion >= 0x090000)
		{
			arguments << "--format=legacy";
		}
        arguments << parseGeneralOptions(options, false, true);
        m_pipProcess.start(m_pythonPath, arguments);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::listAvailablePackages2(const QStringList &names)
{
    if (m_pythonPath == "")
    {
        emit outputAvailable("Python is not available\n", false);
        return;
    }

    if (m_pipAvailable == false)
    {
        emit outputAvailable("pip is not available\n", false);
        return;
    }

    //list consists of two steps:
    //1. get package names using freeze
    //2. get more information using show package1 package2 ...
    if (m_currentTask == taskNo)
    {
        emit pipRequestStarted(taskListPackages2, "Get list of installed packages... (step 2)\n", true);
        clearBuffers();
        m_currentTask = taskListPackages2;

        QStringList arguments;
        arguments << "-m" << "pip" << "show" << names;
        arguments << parseGeneralOptions(m_generalOptionsCache, false, true); //version has already been checked in listAvailablePackages. This is sufficient.
        m_pipProcess.start(m_pythonPath, arguments);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::checkPackageUpdates(const PipGeneralOptions &options /*= PipGeneralOptions()*/)
{
    if (m_pythonPath == "")
    {
        emit outputAvailable("Python is not available\n", false);
        return;
    }

    if (m_pipAvailable == false)
    {
        emit outputAvailable("pip is not available\n", false);
        return;
    }

    if (m_currentTask == taskNo)
    {
        emit pipRequestStarted(taskCheckUpdates, "Check online (pypi.python.org) if newer versions of packages are available...\n");
        clearBuffers();
        m_currentTask = taskCheckUpdates;

        QStringList arguments;
        arguments << "-m" << "pip" << "list" << "--outdated"; //version has already been checked in listAvailablePackages. This is sufficient.
        if (m_pipVersion >= 0x120000) // >= 18.0
        {
            arguments << "--format=columns";
        }
        else if (m_pipVersion >= 0x090000)
        {
            arguments << "--format=legacy";
        }
        arguments << parseGeneralOptions(options, false, true);
        m_pipProcess.start(m_pythonPath, arguments);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::checkVerifyInstalledPackages(const PipGeneralOptions &options /*= PipGeneralOptions()*/)
{
    if (m_pythonPath == "")
    {
        emit outputAvailable("Python is not available\n", false);
        return;
    }

    if (m_pipAvailable == false)
    {
        emit outputAvailable("pip is not available\n", false);
        return;
    }

    if (m_currentTask == taskNo)
    {
        emit pipRequestStarted(taskVerifyInstalledPackages, "Verify installed packages have compatible dependencies...\n");
        clearBuffers();
        m_currentTask = taskVerifyInstalledPackages;

        QStringList arguments;
        arguments << "-m" << "pip" << "check"; //version has already been checked in listAvailablePackages. This is sufficient.
        arguments << parseGeneralOptions(options, false, true);
        m_pipProcess.start(m_pythonPath, arguments);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::installPackage(const PipInstall &installSettings, const PipGeneralOptions &options /*= PipGeneralOptions()*/)
{
    if (m_pythonPath == "")
    {
        emit outputAvailable("Python is not available\n", false);
        return;
    }

    if (m_pipAvailable == false)
    {
        emit outputAvailable("pip is not available\n", false);
        return;
    }

    if (m_currentTask == taskNo)
    {
        emit pipRequestStarted(taskInstall, "Install package...\n");
        clearBuffers();
        m_currentTask = taskInstall;

        QStringList arguments;
        arguments << "-m" << "pip" << "install";

        if (installSettings.type != PipInstall::typeRequirements)
        {
            if (installSettings.upgrade)
            {
                arguments << "--upgrade";
            }

            if (!installSettings.installDeps)
            {
                arguments << "--no-deps";
            }
        }

        if (installSettings.ignoreIndex)
        {
            arguments << "--no-index";
        }

        if (installSettings.findLinks != "")
        {
            arguments << "--find-links" << installSettings.findLinks;
        }

        if (installSettings.type == PipInstall::typeWhl)
        {
            if (m_pipVersion >= 0x070100)
            {
                arguments << "--prefer-binary"; // << ("--only-binary=" + installSettings.packageName.trimmed());
            }
        }
        else if (installSettings.type == PipInstall::typeSearchIndex)
        {
            if (m_pipVersion >= 0x070100)
            {
                arguments << "--prefer-binary";
            }
        }
        else if (installSettings.type == PipInstall::typeTarGz) // typeTarGz
        {
            if (m_pipVersion >= 0x070100)
            {
                arguments << "--prefer-binary"; // << ("--no-binary=" + installSettings.packageName.trimmed());
            }
        }

        arguments << parseGeneralOptions(options, false, true); //version has already been checked in listAvailablePackages. This is sufficient.

        if (installSettings.type == PipInstall::typeRequirements) // typeRequirements
        {
            arguments << "-r";
        }
        
        arguments << installSettings.packageName;

        emit pipRequestStarted(taskInstall, arguments.mid(1).join(" ") + "\n");

        if (installSettings.runAsSudo)
        {
            arguments.push_front(m_pythonPath);
            m_pipProcess.start("pkexec", arguments);
        }
        else
        {
            m_pipProcess.start(m_pythonPath, arguments);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::uninstallPackage(const QString &packageName, bool runAsSudo, const PipGeneralOptions &options /*= PipGeneralOptions()*/)
{
    if (m_pythonPath == "")
    {
        emit outputAvailable("Python is not available\n", false);
        return;
    }

    if (m_pipAvailable == false)
    {
        emit outputAvailable("pip is not available\n", false);
        return;
    }

    if (m_currentTask == taskNo)
    {
        emit pipRequestStarted(taskUninstall, QString("Uninstall package %1...\n").arg(packageName));
        clearBuffers();
        m_currentTask = taskUninstall;

        QStringList arguments;
        arguments << "-m" << "pip" << "uninstall" << "--yes"; //version has already been checked in listAvailablePackages. This is sufficient.

        arguments << parseGeneralOptions(options, false, true);

        arguments << packageName;

        if (runAsSudo)
        {
            arguments.push_front(m_pythonPath);
            m_pipProcess.start("pkexec", arguments);
        }
        else
        {
            m_pipProcess.start(m_pythonPath, arguments);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::processError(QProcess::ProcessError error)
{
    if (m_currentTask != taskNo)
    {
        switch (error)
        {
        case QProcess::FailedToStart:
            emit outputAvailable(tr("Could not start python pip\n"), false);
            break;
        case QProcess::ReadError:
            emit outputAvailable(tr("An error occurred when attempting to read from the process.\n"), false);
            break;
        default:
            emit outputAvailable(tr("other error"), false);
            break;
        }
    }

    finalizeTask();
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::processFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if (exitStatus == QProcess::CrashExit)
    {
        Task temp = m_currentTask;
        m_currentTask = taskNo;

        emit pipRequestFinished(temp, tr("Python pip crashed during execution\n"), false);

        if (temp != taskNo)
        {
            processReadyReadStandardError();
            processReadyReadStandardOutput();
        }

        clearBuffers();
    }
    else
    {
        qDebug() << "pip exit code:" << m_pipProcess.exitCode();
        finalizeTask(m_pipProcess.exitCode());
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::processReadyReadStandardError()
{
    QByteArray str = m_pipProcess.readAllStandardError();
    if (str.length() > 0)
    {
        m_standardErrorBuffer += str;
        emit outputAvailable(str, false);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::processReadyReadStandardOutput()
{
    QByteArray str = m_pipProcess.readAllStandardOutput();
    if (str.length() > 0)
    {
        m_standardOutputBuffer += str;
        emit outputAvailable(str, true);
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::finalizeTask(int exitCode /*= 0*/)
{
    Task temp = m_currentTask;
    m_currentTask = taskNo;

    if (temp != taskNo)
    {
        processReadyReadStandardError();
        processReadyReadStandardOutput();

        QString error = m_standardErrorBuffer;
        QString output = m_standardOutputBuffer;

        if (temp == taskCheckAvailable)
        {
            if (exitCode == 0)
            {
                QRegExp reg("pip ((\\d+)\\.(\\d+)\\.(\\d+)) from(.*)");
                if (reg.indexIn(output) != -1)
                {
                    m_pipVersion = CREATEVERSION(reg.cap(2).toInt(), reg.cap(3).toInt(), reg.cap(4).toInt());
                    QString version = reg.cap(1);
                    emit pipVersion(version);
                    m_pipAvailable = true;
                    emit pipRequestFinished(temp, "", true);
                }
                else
                {
                    QRegExp reg("pip ((\\d+)\\.(\\d+)) from(.*)");
                    if (reg.indexIn(output) != -1)
                    {
                        m_pipVersion = CREATEVERSION(reg.cap(2).toInt(), reg.cap(3).toInt(), 0);
                        QString version = reg.cap(1);
                        emit pipVersion(version);
                        m_pipAvailable = true;
                        emit pipRequestFinished(temp, "", true);
                    }
                    else
                    {
                        m_pipAvailable = false;
                        emit pipRequestFinished(temp, "Package pip is not available. Install Python pip first (see https://pip.pypa.io/en/latest/installing.html).\n", false);
                    }
                }
            }
            else if (exitCode == 3)
            {
                m_pipAvailable = false;
                emit pipRequestFinished(temp, "Python returned with the error code 3 (no such process). Possibly, the PYTHONHOME environment variable or the corresponding setting in the property dialog of itom is not correctly set to the base directory of Python. Please correct this.", false);
            }
            else
            {
                m_pipAvailable = false;
                emit pipRequestFinished(temp, QString("Python returned with the exit code %1. Please see the module 'errno' for error codes.").arg(exitCode), false);
            }
        }
        else if (temp == taskListPackages1)
        {
            if (error != "" && output == "")
            {
                emit pipRequestFinished(temp, "Error obtaining list of packages (list)\n", false);
            }
            else
            {
                QStringList packages_out;
                int idx;
                QStringList packages = output.split("\n");

                if (m_pipVersion >= 0x120000) //>= 18.0
                {
                    //format columns (first line are headings, then one line with dashes)
                    for (int idx = 2; idx < packages.size(); ++idx)
                    {
                        QStringList items = packages[idx].split(" ");
                        if (items.size() > 0)
                        {
                            packages_out.append(items[0]);
                        }
                    }
                }
                else
                {
                    //format legacy
                    foreach(const QString &p, packages)
                    {
                        idx = p.indexOf(" (");
                        if (idx != -1)
                        {
                            packages_out.append(p.left(idx));
                        }
                    }
                }

                //in rare cases, there are temporary, backup directories, starting with '-'.
                //they have to be removed here.
                QStringList::iterator it = packages_out.begin();

                while (it != packages_out.end())
                {
                    if (it->startsWith("-"))
                    {
                        it = packages_out.erase(it);
                    }
                    else
                    {
                        it++;
                    }
                }

                if (packages_out.length() > 0)
                {
                    listAvailablePackages2(packages_out);
                }
            }
        }
        else if (temp == taskListPackages2)
        {
            if (error != "" && output == "")
            {
                emit pipRequestFinished(temp, "Error obtaining list of packages (show)\n", false);
            }
            else
            {
                beginResetModel();
                m_pythonPackages.clear();

                QStringList lines = output.split("\r\n");
                if (lines.length() == 1) //nothing found (e.g. older pip or linux)
                {
                    lines = output.split("\n");
                }

				//The "python.exe - m pip show numpy pip setuptools" request returns a stream in the following way:
				/*
				---
				Name: numpy
				Version: 1.11.0
				Summary: NumPy: array processing for numbers, strings, records, and objects.
				Home-page: http://www.numpy.org
				Author: NumPy Developers
				Author-email: numpy-discussion@scipy.org
				License: BSD
				Location: c:\program files\python35\lib\site-packages
				Requires:
				---
				Name: pip
				Version: 9.0.1
				Summary: The PyPA recommended tool for installing Python packages.
				Home-page: https://pip.pypa.io/
				Author: The pip developers
				Author-email: python-virtualenv@groups.google.com
				License: MIT
				Location: c:\program files\python35\lib\site-packages
				Requires:
				---
				Name: setuptools
				Version: 18.2
				Summary: Easily download, build, install, upgrade, and uninstall Python packages

				Home-page: https://bitbucket.org/pypa/setuptools
				Author: Python Packaging Authority
				Author-email: distutils-sig@python.org
				License: PSF or ZPL
				Location: c:\program files\python35\lib\site-packages
				Requires:
				*/

				//The following code puts every package into the PythonPackage struct.
				//Once the next ---line is found, the previous package struct is appended to m_pythonPackages
				//and a new package struct is created.

				//Starting from pip 9.0.0, the response does not start with a --- line, therefore
				// package_started has to be set to true in this case, while it was false for pip < 9.0.0

                PythonPackage package;
                bool package_started = false;
				if (m_pipVersion >= 0x090000)
				{
					package_started = true;
				}
                int pos;
                QString key, value;
                QStringList keys;
                keys << "Name" << "Version" << "Summary" << "Home-page" << "License" << "Location" << "Requires";

                foreach (const QString &line, lines)
                {
                    if (line == "---")
                    {
                        if (package_started)
                        {
                            m_pythonPackages << package;
                        }

                        package_started = true;
                        package = PythonPackage(); //start new, empty package structure
                    }
                    else if (line != "")
                    {
                        //check if line consists of key: value
                        pos = line.indexOf(": ");
                        if (pos != -1)
                        {
                            key = line.left(pos);
                            value = line.mid(pos+2);

                            switch (keys.indexOf(key))
                            {
                            case 0: //Name
                                package.m_name = value;
                                break;
                            case 1: //Version
                                package.m_version = value;
                                break;
                            case 2: //Summary
                                package.m_summary = value;
                                break;
                            case 3: //Home-page
                                package.m_homepage = value;
                                break;
                            case 4: //License
                                package.m_license = value;
                                break;
                            case 5: //Location
                                package.m_location = value;
                                break;
                            case 6: //Requires
                                package.m_requires = value;
                                break;
                            }
                        }
                    }
                }

                if (package_started)
                {
                    m_pythonPackages << package;
                }

                endResetModel();
                emit pipRequestFinished(temp, "List of packages obtained.\n", true);
            }
        }
        else if (temp == taskCheckUpdates)
        {
            if (error != "" && output == "")
            {
                emit pipRequestFinished(temp, "Error obtaining list of outdated packages (list)\n", false);
            }
            else
            {
                QMap<QString, QString> outdated;
                QMap<QString, QString> unknown;

                if (m_pipVersion >= 0x120000)
                {
                    QStringList lines = output.split("\n");
                    for (int idx = 2; idx < lines.size(); ++idx)
                    {
                        QStringList items = lines[idx].split(QRegExp("\\s+"), QString::SkipEmptyParts);
                        if (items.size() >= 4)
                        {
                            outdated[items[0]] = items[2];
                        }
                    }
                }
                else
                {
                    QRegExp rx("(\\S+) \\(Current: (\\S+) Latest: (\\S+)( \\[\\S+\\])?\\)"); //the style is "scipy (Current: 0.16.1 Latest: 0.17.0 [sdist])"
                    int pos = 0;
                    
                    while ((pos = rx.indexIn(output, pos)) != -1)
                    {
                        outdated[rx.cap(1)] = rx.cap(3);
                        pos += rx.matchedLength();
                    }

                    //check for style of pip >= 8.0.0
                    pos = 0;
                    rx.setPattern("(\\S+) \\((\\S+)(, \\S+)?\\) - Latest: (\\S+)( \\[\\S+\\])?"); //the style is "scipy (0.16.1) - Latest: 0.17.0 [sdist]" or "scipy (0.16.1, path-to-location) - Latest: 0.17.0 [sdist]"
                    while ((pos = rx.indexIn(output, pos)) != -1)
                    {
                        outdated[rx.cap(1)] = rx.cap(4);
                        pos += rx.matchedLength();
                    }

                    //check for unknown (that could not been fetched)
                    pos = 0;
                    rx.setPattern("Could not find any downloads that satisfy the requirement (\\S+)");

                    while ((pos = rx.indexIn(output, pos)) != -1)
                    {
                        unknown[rx.cap(1)] = "unknown";
                        pos += rx.matchedLength();
                    }
                }

                for (int i = 0; i < m_pythonPackages.length(); ++i)
                {
                    if (outdated.contains(m_pythonPackages[i].m_name))
                    {
                        m_pythonPackages[i].m_newVersion = outdated[m_pythonPackages[i].m_name];
                        m_pythonPackages[i].m_status = PythonPackage::Outdated;
                    }
                    else if (unknown.contains(m_pythonPackages[i].m_name))
                    {
                      m_pythonPackages[i].m_status = PythonPackage::Unknown;
                    }
                    else
                    {
                        m_pythonPackages[i].m_status = PythonPackage::Uptodate;
                    }
                }

                emit dataChanged(createIndex(0,4), createIndex(m_pythonPackages.length()-1, 4));

                emit pipRequestFinished(temp, "Packages checked.\n", true);
            }
        }
        else if (temp == taskVerifyInstalledPackages)
        {
            if (error != "" && output == "")
            {
                emit pipRequestFinished(temp, "Error verifying if installed packages have compatible dependencies. (check)\n", false);
            }
            else
            {
                emit pipRequestFinished(temp, "Finished.\n", true);
            }
        }
        else if (temp == taskInstall)
        {
            if (error != "" && output == "")
            {
                emit pipRequestFinished(temp, "Error installing package\n", false);
            }
            else
            {
                listAvailablePackages();
            }
        }
        else if (temp == taskUninstall)
        {
            if (error != "" && output == "")
            {
                emit pipRequestFinished(temp, "Error uninstalling package\n", false);
            }
            else
            {
                listAvailablePackages();
            }
        }
    }

    clearBuffers();
}

//-----------------------------------------------------------------------------------------
QStringList PipManager::parseGeneralOptions(const PipGeneralOptions &options, bool ignoreRetries /*= false*/, bool ignoreVersionCheck /*= true*/) const
{
    QStringList output;

    if (options.isolated)
    {   
        output << "--isolated";
    }

    if (options.logPath != "")
    {
        output << "--log" << QString("\"%1\"").arg(options.logPath);
    }

    if (options.proxy != "")
    {
        output << "--proxy" << options.proxy;
    }

    if (options.timeout >= 0)
    {
        output << "--timeout" << QString("%1").arg(options.timeout);
    }

    if (options.retries > 0 && !ignoreRetries && MAJORVERSION(m_pipVersion) >= 6)
    {
        output << "--retries" << QString("%1").arg(options.retries);
    }

    if (options.useTrustedHosts && MAJORVERSION(m_pipVersion) >= 6)
    {
        foreach(const QString &th, options.trustedHosts)
        {
            output << QString("--trusted-host=%1").arg(th);
        }
    }

    if (ignoreVersionCheck && MAJORVERSION(m_pipVersion) >= 6)
    {
        output << "--disable-pip-version-check";
    }

    return output;
}

//-----------------------------------------------------------------------------------------
void PipManager::clearBuffers()
{
    m_standardOutputBuffer.clear();
    m_standardErrorBuffer.clear();
}

//-----------------------------------------------------------------------------------------
void PipManager::interruptPipProcess()
{
    if (m_pipProcess.state() == QProcess::Running || m_pipProcess.state() == QProcess::Starting)
    {
        m_pipProcess.kill();
    }
}

//-----------------------------------------------------------------------------------------
bool PipManager::isPackageInUseByOther(const QModelIndex &index)
{
    if (!index.isValid() || index.row() < 0 || index.row() >= m_pythonPackages.size())
    {
        return false;
    }

    QString other = m_pythonPackages[index.row()].m_name;
    QStringList requires;
    foreach (const PythonPackage &pp, m_pythonPackages)
    {
        requires = pp.m_requires.split(", ");
        if (requires.contains(other, Qt::CaseInsensitive))
        {
            return true;
        }
    }

    return false;
}

} //end namespace ito