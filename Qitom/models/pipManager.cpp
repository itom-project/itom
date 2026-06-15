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
#include <QRegularExpression>

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
        m_pUserDefinedPythonHome(nullptr),
        m_pipCallMode(PipMode::pipModeDirect),
        m_numberOfUnfetchedPackageDetails(0),
        m_numberOfNewlyObtainedPackageDetails(0),
        m_fetchDetailCancelRequested(false)
    {
        m_headers << tr("Name") << tr("Version") << tr("Location") << tr("Requires") << tr("Updates") << tr("Summary") << tr("Homepage") << tr("License");
        m_alignment << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft) << QVariant(Qt::AlignLeft);

        connect(&m_pipProcess, &QProcess::errorOccurred, this, &PipManager::processError);

#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
        connect(
            &m_pipProcess,
            SIGNAL(finished(int, QProcess::ExitStatus)),
            this,
            SLOT(processFinished(int, QProcess::ExitStatus)));
#else
        connect(&m_pipProcess, &QProcess::finished, this, &PipManager::processFinished);
#endif
        connect(&m_pipProcess, &QProcess::readyReadStandardError, this, &PipManager::processReadyReadStandardError);
        connect(&m_pipProcess, &QProcess::readyReadStandardOutput, this, &PipManager::processReadyReadStandardOutput);

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

        if (!retval.containsError())
        {
            checkCallMode();
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
        const QString pythonExe = QString("/python%1%2.dll").arg(PY_MAJOR_VERSION).arg(PY_MINOR_VERSION);

        if (QDir(pythonSubDir).exists() && \
            QFileInfo(pythonSubDir + pythonExe).exists())
        {
            pythonDirState = 0; //use pythonXX subdirectory of itom as python home path
        }
        else if (QDir(pythonAllInOneDir).exists() && \
            QFileInfo(pythonAllInOneDir + pythonExe).exists())
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

#if (PY_VERSION_HEX >= 0x030B0000)
    // from Python 3.11 on, the pre-init config is used to
    // configure Python to a local configuration with UTF8 mode.
    PyStatus status;
    PyPreConfig preconfig;

    PyPreConfig_InitIsolatedConfig(&preconfig);

    preconfig.utf8_mode = 1;
    preconfig.use_environment = 0;
    preconfig.parse_argv = 0;

    status = Py_PreInitialize(&preconfig);

    if (PyStatus_Exception(status))
    {
        return RetVal::format(retError, 0, tr("Error pre-initializing Python in isolated mode:  %s").toLatin1().data(),
            status.err_msg);
    }
#endif

    if (pythonDir != "")
    {
        //the python home path given to Py_SetPythonHome must be persistent for the whole Python session
        m_pUserDefinedPythonHome = Py_DecodeLocale(pythonDir.toLatin1().data(), NULL);
    }

#if (PY_VERSION_HEX >= 0x030B0000)
    // from Python 3.11 on, the init config is used to
    // configure Python.
    PyConfig config;

    PyConfig_InitIsolatedConfig(&config);

    /* Set the program name before reading the configuration
       (decode byte string from the locale encoding).

       Implicitly preinitialize Python. */
       /*status = PyConfig_SetBytesString(&config, &config.program_name,
           program_name);
       if (PyStatus_Exception(status)) {
           goto done;
       }*/

    if (m_pUserDefinedPythonHome)
    {
        status = PyConfig_SetString(&config, &config.home, m_pUserDefinedPythonHome);
        if (PyStatus_Exception(status))
        {
            PyConfig_Clear(&config);
            return RetVal::format(retError, 0, tr("Error setting custom Python home path:  %s").toLatin1().data(),
                status.err_msg);
        }
    }

    /* Read all configuration at once */
    status = PyConfig_Read(&config);

    if (PyStatus_Exception(status))
    {
        PyConfig_Clear(&config);
        return RetVal::format(retError, 0, tr("Error reading the Python configuration: %s").toLatin1().data(),
            status.err_msg);
    }

    status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);

    if (PyStatus_Exception(status))
    {
        return RetVal::format(retError, 0, tr("Error initializing Python: %s.\nVerify the Python base directory in the itom property dialog and restart itom.").toLatin1().data(),
            status.err_msg);
    }

#else
    if (m_pUserDefinedPythonHome)
    {
        Py_SetPythonHome(m_pUserDefinedPythonHome);
    }

    //!< must be called after any PyImport_AppendInittab-call
    Py_Initialize();
#endif

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
        retval += RetVal::format(
            retError,
            0,
            tr("The home directory of Python is currently set to the non-existing directory '%s'\nPython cannot be started. Please set either the environment variable PYTHONHOME to the base directory of python \nor correct the base directory in the property dialog of itom.").toLatin1().data(),
            pythonHomeDir.absolutePath().toLatin1().data()
        );

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
        retval += RetVal::format(
            retError,
            0,
            tr("The built-in library path of Python could not be found. The current home directory is '%s'\nPython cannot be started. Please set either the environment variable PYTHONHOME to the base directory of python \nor correct the base directory in the preferences dialog of itom.").toLatin1().data(),
            pythonHomeDir.absolutePath().toLatin1().data()
        );

        return retval;
    }

    return retval;
}

//-------------------------------------------------------------------------------------
/* There are some packages, whose meta information contain special characters.
It turned out, that QProcess uses the default Windows encoding cp1252 under Windows
to communicate via std::cout and std::cerr with the called process.
This encoding can for instance not decode the latin letter L with Stroke (unicode 141).
Then pip raises an UnicodeEncodeError.

This problem seems only exist under Windows. If a Python script is called via QProcess
the sys.stdout.encoding and sys.stderr.encoding is usually set to 'cp1252', however
we wish to have 'utf8' as encoding. Therefore we have an alternative approach to
indirectly call pip via the itom module itom-packages/pipProcess/runPipUtf8.

This module at first reconfigures the encoding of cout and cerr streams to 'utf8' and
then calls pip via a non-official approach, documented in pip/_internal/cli/main.py,
using runpy. Since this approach might change for different pip versions, this
method silently checks this call using a simple example. If it succeeds, runPipUtf8.py
is used for all calls to pip, else the direct call is used.
*/
ito::RetVal PipManager::checkCallMode()
{
#ifdef WIN32
    QProcess process;
    QStringList args;
    QDir pipProcessDir = QCoreApplication::applicationDirPath();

    pipProcessDir.cd("itom-packages");
    pipProcessDir.cd("pipProcess");

    QFileInfo runPipUtf8(pipProcessDir, "runPipUtf8.py");

    if (runPipUtf8.exists())
    {
        m_runPipUtf8Path = runPipUtf8.absoluteFilePath();
        args << m_runPipUtf8Path << "pip" << "-V";  // get version
        int exitCode = QProcess::execute(m_pythonPath, args); // 0: ok, 1: any error

        if (exitCode == 0)
        {
            m_pipCallMode = PipMode::pipModeRunPipUtf8;
        }
    }

#endif

    return ito::retOk;
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
    if (parent.isValid() || row < 0 || row >= m_pythonPackages.length() || column < 0 || column >= m_headers.size())
    {
        return QModelIndex();
    }

    return createIndex(row, column);
}

//----------------------------------------------------------------------------------------------------------------------------------
bool PipManager::isPipStarted() const
{
    return m_pipProcess.processId() != 0;
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::startProcess(const QStringList &arguments)
{
    m_fetchDetailCancelRequested = false;

    if (m_pipCallMode == PipMode::pipModeDirect)
    {
        QStringList args;
        args << "-m" << arguments;
        m_pipProcess.start(m_pythonPath, args);
    }
    else
    {
        QStringList args;
        args << m_runPipUtf8Path << arguments;
        m_pipProcess.start(m_pythonPath, args);
    }
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
        if (PY_VERSION_HEX >= 0x03050000 && PY_VERSION_HEX < 0x03090000)
        {
            emit pipRequestStarted(taskCheckAvailable,
                "For Python 3.5 or higher, some packages (e.g. Scipy or OpenCV) \
might depend on the Microsoft Visual C++ 2015 redistributable package. Please \
install it if not yet done.\n\nCheck connection to pip and get version...\n");
        }
        else
        {
            emit pipRequestStarted(taskCheckAvailable,
                QString("Python %1.\nCheck connection to pip and get version...\n").arg(PY_VERSION)
            );
        }
#else
        emit pipRequestStarted(taskCheckAvailable,
            QString("Python %1.\nCheck connection to pip and get version...\n").arg(PY_VERSION)
        );
#endif

        clearBuffers();
        m_currentTask = taskCheckAvailable;

        QStringList arguments;
        arguments << "-m" << "pip" << "-V";
        arguments << parseGeneralOptions(options, true);
        m_pipProcess.start(m_pythonPath, arguments); // always use the direct call to pip here
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void PipManager::listAvailablePackages(const PipGeneralOptions &options /*= PipGeneralOptions()*/, bool forceReloadDetails /*= false*/)
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
        emit pipRequestStarted(taskListPackages, "Step 1: Get list of names and versions of installed packages...\n", true);
        clearBuffers();
        m_currentTask = taskListPackages;
        m_generalOptionsCache = options;

        if (forceReloadDetails)
        {
            beginResetModel();
            m_pythonPackages.clear();
            endResetModel();
        }

        QStringList arguments;
        arguments << "pip" << "list"; //here the pip version check is done

        if (m_pipVersion >= 0x120000) // >= 18.0
        {
            arguments << "--format=columns";
        }
        else if (m_pipVersion >= 0x090000)
        {
            arguments << "--format=legacy";
        }

        arguments << parseGeneralOptions(options, false, true);
        startProcess(arguments);
    }
}

//-------------------------------------------------------------------------------------
void PipManager::fetchPackageDetails(const QStringList &names, int totalNumberOfUnfetchedDetails, bool firstCall)
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
        QString text;

        if (firstCall)
        {
            text = QString("Step 2: Fetch details of %1 out of %2 installed packages...\n").arg(totalNumberOfUnfetchedDetails).arg(m_pythonPackages.size());
            m_numberOfUnfetchedPackageDetails = totalNumberOfUnfetchedDetails;
            m_numberOfNewlyObtainedPackageDetails = 0;
            emit pipFetchDetailsProgress(totalNumberOfUnfetchedDetails, 0, false);
        }

        emit pipRequestStarted(taskFetchPackagesDetails, text, true);
        clearBuffers();
        m_currentTask = taskFetchPackagesDetails;

        QStringList arguments;
        arguments << "pip" << "show" << names;
        arguments << parseGeneralOptions(m_generalOptionsCache, false, true); //version has already been checked in listAvailablePackages. This is sufficient.
        startProcess(arguments);
    }
}

//-------------------------------------------------------------------------------------
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
        arguments << "pip" << "list" << "--outdated"; //version has already been checked in listAvailablePackages. This is sufficient.

        if (m_pipVersion >= 0x120000) // >= 18.0
        {
            arguments << "--format=columns";
        }
        else if (m_pipVersion >= 0x090000)
        {
            arguments << "--format=legacy";
        }

        arguments << parseGeneralOptions(options, false, true);
        startProcess(arguments);
    }
}

//-------------------------------------------------------------------------------------
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
        arguments << "pip" << "check"; //version has already been checked in listAvailablePackages. This is sufficient.
        arguments << parseGeneralOptions(options, false, true);
        startProcess(arguments);
    }
}

//-------------------------------------------------------------------------------------
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
        arguments << "pip" << "install";

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
            if (m_pipVersion >= 0x120000) // version >= 18.0
            {
                arguments << "--prefer-binary"; // << ("--only-binary=" + installSettings.packageName.trimmed());
            }
        }
        else if (installSettings.type == PipInstall::typeSearchIndex)
        {
            if (m_pipVersion >= 0x120000) // version >= 18.0
            {
                arguments << "--prefer-binary";
            }
        }
        else if (installSettings.type == PipInstall::typeTarGz) // typeTarGz
        {
            if (m_pipVersion >= 0x120000) // version >= 18.0
            {
                arguments << "--prefer-binary"; // << ("--no-binary=" + installSettings.packageName.trimmed());
            }
        }

        arguments << parseGeneralOptions(options, false, true); //version has already been checked in listAvailablePackages. This is sufficient.

        if (installSettings.type == PipInstall::typeRequirements) // typeRequirements
        {
            arguments << "-r";
        }
        else if (installSettings.type == PipInstall::typePackageSource) // pip development mode of python packages
        {
            arguments << "-e";
        }


        // if typeSearchIndex, multiple packages can be installed. They
        // are separated by spaces

        if (installSettings.type == PipInstall::typeSearchIndex)
        {
            auto packageNames = installSettings.packageName.split(" ");

            foreach (const QString& pn, packageNames)
            {
                if (pn.trimmed() != "")
                {
                    arguments << pn.trimmed();
                }
            }
        }
        else
        {
            arguments << installSettings.packageName;
        }

        emit pipRequestStarted(taskInstall, arguments.mid(1).join(" ") + "\n");

        if (installSettings.runAsSudo)
        {
            arguments.push_front("-m");
            arguments.push_front(m_pythonPath);
            m_pipProcess.start("pkexec", arguments);
        }
        else
        {
            startProcess(arguments);
        }
    }
}

//-------------------------------------------------------------------------------------
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
        arguments << "pip" << "uninstall" << "--yes"; //version has already been checked in listAvailablePackages. This is sufficient.

        arguments << parseGeneralOptions(options, false, true);

        arguments << packageName;

        if (runAsSudo)
        {
            arguments.push_front("-m");
            arguments.push_front(m_pythonPath);
            m_pipProcess.start("pkexec", arguments);
        }
        else
        {
            startProcess(arguments);
        }
    }
}

//-------------------------------------------------------------------------------------
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
        case QProcess::Crashed:
            if (!m_fetchDetailCancelRequested)
            {
                emit outputAvailable(tr("Process crashed.\n"), false);
            }
            break;
        default:
            emit outputAvailable(tr("other error\n"), false);
            break;
        }

        if (m_currentTask == taskFetchPackagesDetails)
        {
            emit pipFetchDetailsProgress(m_numberOfUnfetchedPackageDetails, m_numberOfUnfetchedPackageDetails, true);
        }
    }

    finalizeTask();
}

//-------------------------------------------------------------------------------------
void PipManager::processFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if (exitStatus == QProcess::CrashExit)
    {
        Task temp = m_currentTask;
        m_currentTask = taskNo;

        if (!m_fetchDetailCancelRequested)
        {
            emit pipRequestFinished(temp, tr("Python pip crashed during execution\n"), false);
        }

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

//-------------------------------------------------------------------------------------
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
            finalizeTaskCheckAvailable(error, output, exitCode);
        }
        else if (temp == taskListPackages)
        {
            finalizeTaskListPackages(error, output);
        }
        else if (temp == taskFetchPackagesDetails)
        {
            finalizeTaskFetchPackagesDetails(error, output);
        }
        else if (temp == taskCheckUpdates)
        {
            finalizeTaskCheckUpdates(error, output);
        }
        else if (temp == taskVerifyInstalledPackages)
        {
            finalizeTaskVerifyInstalledPackages(error, output);
        }
        else if (temp == taskInstall)
        {
            finalizeTaskInstall(error, output);
        }
        else if (temp == taskUninstall)
        {
            finalizeTaskUninstall(error, output);
        }
    }

    clearBuffers();
}

//-------------------------------------------------------------------------------------
void PipManager::finalizeTaskCheckAvailable(const QString& error, const QString& output, int exitCode)
{
    if (exitCode == 0)
    {
        QRegularExpression reg("pip ((\\d+)\\.(\\d+)\\.(\\d+)) from(.*)");
        QRegularExpressionMatch match = reg.match(output);
        if (match.hasMatch())
        {
            m_pipVersion = CREATEVERSION(
                match.captured(2).toInt(),
                match.captured(3).toInt(),
                match.captured(4).toInt());
            QString version = match.captured(1);
            emit pipVersion(version);
            m_pipAvailable = true;
            emit pipRequestFinished(taskCheckAvailable, "", true);
        }
        else
        {
            QRegularExpression reg("pip ((\\d+)\\.(\\d+)) from(.*)");
            QRegularExpressionMatch match = reg.match(output);
            if (match.hasMatch())
            {
                m_pipVersion = CREATEVERSION(match.captured(2).toInt(), match.captured(3).toInt(), 0);
                QString version = match.captured(1);
                emit pipVersion(version);
                m_pipAvailable = true;
                emit pipRequestFinished(taskCheckAvailable, "", true);
            }
            else
            {
                m_pipAvailable = false;
                emit pipRequestFinished(taskCheckAvailable, "Package pip is not available. Install Python pip first (see https://pip.pypa.io/en/latest/installing.html).\n", false);
            }
        }
    }
    else if (exitCode == 3)
    {
        m_pipAvailable = false;
        emit pipRequestFinished(taskCheckAvailable, "Python returned with the error code 3 (no such process). Possibly, the PYTHONHOME environment variable or the corresponding setting in the property dialog of itom is not correctly set to the base directory of Python. Please correct this.", false);
    }
    else
    {
        m_pipAvailable = false;
        emit pipRequestFinished(taskCheckAvailable, QString("Python returned with the exit code %1. Please see the module 'errno' for error codes.").arg(exitCode), false);
    }
}

//-------------------------------------------------------------------------------------
void PipManager::finalizeTaskListPackages(const QString& error, const QString& output)
{
    if (error != "" && output == "")
    {
        emit pipRequestFinished(taskListPackages, "Error obtaining list of packages (list)\n", false);
    }
    else
    {
        int idx;
        QStringList packages = output.split("\n");

        // map of new packages (name -> version, version can also be an empty string for older pip versions)
        QMap<QString, QString> listedPackages;

        beginResetModel();

        if (m_pipVersion >= 0x120000) //>= 18.0
        {
            //format columns (first line are headings, then one line with dashes)
            for (int idx = 2; idx < packages.size(); ++idx)
            {
                QStringList items = packages[idx].split(QRegularExpression("\\s+"));

                if (items.size() > 0 && items[0].trimmed() != "")
                {
                    listedPackages[items[0].trimmed()] = items[1].trimmed();
                }
            }
        }
        else
        {
            //format legacy
            foreach(const QString & p, packages)
            {
                idx = p.indexOf(" (");

                if (idx != -1)
                {
                    listedPackages[p.left(idx)] = ""; // no version tag
                }
            }
        }

        // remove all packages from current list that are not in listedPackages any more
        // or update an existing item by the new version
        for (auto it = m_pythonPackages.begin(); it != m_pythonPackages.end(); /* noop */)
        {
            if (!listedPackages.contains(it->m_name))
            {
                // package does not exist any more
                it = m_pythonPackages.erase(it);
            }
            else
            {
                if (listedPackages[it->m_name] != it->m_version)
                {
                    // version has changed -> update version and reset details
                    it->m_version = listedPackages[it->m_name];
                    it->m_homepage = "";
                    it->m_detailsFetched = false;
                    it->m_license = "";
                    it->m_location = "";
                    it->m_newVersion = "";
                    it->m_requires = "";
                    it->m_status = PythonPackage::Unknown;
                    it->m_summary = "";
                }

                // remove package from "new list", since it has been handled
                listedPackages.remove(it->m_name);
                it++;
            }
        }

        // add all new packages to the m_pythonPackages list
        for (auto it = listedPackages.constBegin(); it != listedPackages.constEnd(); it++)
        {
            PythonPackage item(it.key(), it.value());
            item.m_detailsFetched = false;
            m_pythonPackages.append(item);
        }

        // sort packages by name
        std::sort(m_pythonPackages.begin(), m_pythonPackages.end(),
            [](const ito::PythonPackage& a, const ito::PythonPackage& b) -> bool
            {
                return QString::compare(a.m_name, b.m_name, Qt::CaseInsensitive) <= 0;
            }
        );

        endResetModel();

        if (!triggerFetchDetailsForOpenPackages(true))
        {
            emit pipRequestFinished(taskFetchPackagesDetails, "List of packages obtained.\n", true);
        }
    }
}

//-------------------------------------------------------------------------------------
bool PipManager::triggerFetchDetailsForOpenPackages(bool firstCall)
{
    // fetch all details from packages, that have not been updated yet
    QStringList packages_out;

    foreach(const auto & pckg, m_pythonPackages)
    {
        //in rare cases, there are temporary, backup directories, starting with '-'.
        //they have to be removed here.
        if (!pckg.m_detailsFetched && !pckg.m_name.startsWith("-"))
        {
            packages_out << pckg.m_name;
        }
    }

    if (packages_out.size() > 0)
    {
        fetchPackageDetails(packages_out.mid(0, 5), packages_out.size(), firstCall);
        return true;
    }

    // nothing to trigger
    return false;
}

//-------------------------------------------------------------------------------------
void PipManager::updatePythonPackageDetails(const PythonPackage& details)
{
    for (auto it = m_pythonPackages.begin(); it != m_pythonPackages.end(); ++it)
    {
        if (it->m_name == details.m_name)
        {
            *it = details;
            it->m_detailsFetched = true;
        }
    }
}

//-------------------------------------------------------------------------------------
void PipManager::finalizeTaskFetchPackagesDetails(const QString& error, const QString& output)
{
    if (error != "" && output == "")
    {
        emit pipRequestFinished(taskFetchPackagesDetails, "Error obtaining list of packages (show)\n", false);
    }
    else
    {
        beginResetModel();

        QStringList lines = output.split("\r\n");

        if (lines.size() == 1) //nothing found (e.g. older pip or linux)
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

        foreach(const QString & line, lines)
        {
            if (line == "---")
            {
                if (package_started)
                {
                    updatePythonPackageDetails(package);
                    m_numberOfNewlyObtainedPackageDetails++;
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
                    value = line.mid(pos + 2);

                    switch (keys.indexOf(key))
                    {
                    case 0: //Name
                        if (package.m_name == "")
                        {
                            package.m_name = value;
                        }
                        break;
                    case 1: //Version
                        if (package.m_version == "")
                        {
                            package.m_version = value;
                        }
                        break;
                    case 2: //Summary
                        if (package.m_summary == "")
                        {
                            package.m_summary = value;
                        }
                        break;
                    case 3: //Home-page
                        if (package.m_homepage == "")
                        {
                            package.m_homepage = value;
                        }
                        break;
                    case 4: //License
                        if (package.m_license == "")
                        {
                            package.m_license = value;
                        }
                        break;
                    case 5: //Location
                        if (package.m_location == "")
                        {
                            package.m_location = value;
                        }
                        break;
                    case 6: //Requires
                        if (package.m_requires == "")
                        {
                            package.m_requires = value;
                        }
                        break;
                    }
                }
            }
        }

        if (package_started)
        {
            updatePythonPackageDetails(package);
            m_numberOfNewlyObtainedPackageDetails++;
        }

        endResetModel();

        emit pipFetchDetailsProgress(m_numberOfUnfetchedPackageDetails, m_numberOfNewlyObtainedPackageDetails, false);

        if (!m_fetchDetailCancelRequested)
        {
            if (!triggerFetchDetailsForOpenPackages(false))
            {
                emit pipRequestFinished(taskFetchPackagesDetails, "List of packages obtained.\n", true);
                emit pipFetchDetailsProgress(m_numberOfUnfetchedPackageDetails, m_numberOfUnfetchedPackageDetails, true);
            }
        }
        else
        {
            emit pipRequestFinished(taskFetchPackagesDetails, "List of packages obtained (details only partially available).\n", true);
            emit pipFetchDetailsProgress(m_numberOfUnfetchedPackageDetails, m_numberOfUnfetchedPackageDetails, true);
        }
    }
}

//-------------------------------------------------------------------------------------
void PipManager::finalizeTaskCheckUpdates(const QString& error, const QString& output)
{
    if (error != "" && output == "")
    {
        emit pipRequestFinished(taskCheckUpdates, "Error obtaining list of outdated packages (list)\n", false);
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
#if (QT_VERSION >= QT_VERSION_CHECK(5, 14, 0))
                QStringList items = lines[idx].split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
#else
                QStringList items = lines[idx].split(QRegularExpression("\\s+"), QString::SkipEmptyParts);
#endif

                if (items.size() >= 4)
                {
                    outdated[items[0]] = items[2];
                }
            }
        }
        else
        {
            QRegularExpression rx("(\\S+) \\(Current: (\\S+) Latest: (\\S+)( \\[\\S+\\])?\\)"); //the style is "scipy (Current: 0.16.1 Latest: 0.17.0 [sdist])"
            QRegularExpressionMatchIterator matchIterator = rx.globalMatch(output);

            while (matchIterator.hasNext())
            {
                QRegularExpressionMatch match = matchIterator.next();
                outdated[match.captured(1)] = match.captured(3);
            }

            //check for style of pip >= 8.0.0
            rx.setPattern("(\\S+) \\((\\S+)(, \\S+)?\\) - Latest: (\\S+)( \\[\\S+\\])?"); //the style is "scipy (0.16.1) - Latest: 0.17.0 [sdist]" or "scipy (0.16.1, path-to-location) - Latest: 0.17.0 [sdist]"
            matchIterator = rx.globalMatch(output);

            while (matchIterator.hasNext())
            {
                QRegularExpressionMatch match = matchIterator.next();
                outdated[match.captured(1)] = match.captured(4);
            }

            //check for unknown (that could not been fetched)

            rx.setPattern("Could not find any downloads that satisfy the requirement (\\S+)");
            matchIterator = rx.globalMatch(output);

            while (matchIterator.hasNext())
            {
                QRegularExpressionMatch match = matchIterator.next();
                unknown[match.captured(1)] = "unknown";
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

        emit dataChanged(createIndex(0, 4), createIndex(m_pythonPackages.length() - 1, 4));

        emit pipRequestFinished(taskCheckUpdates, "Packages checked.\n", true);
    }
}

//-------------------------------------------------------------------------------------
void PipManager::finalizeTaskVerifyInstalledPackages(const QString& error, const QString& output)
{
    if (error != "" && output == "")
    {
        emit pipRequestFinished(taskVerifyInstalledPackages, "Error verifying if installed packages have compatible dependencies. (check)\n", false);
    }
    else
    {
        emit pipRequestFinished(taskVerifyInstalledPackages, "Finished.\n", true);
    }
}

//-------------------------------------------------------------------------------------
void PipManager::finalizeTaskInstall(const QString& error, const QString& output)
{
    if (error != "" && output == "")
    {
        emit pipRequestFinished(taskInstall, "Error installing package\n", false);
    }
    else
    {
        listAvailablePackages();
    }
}

//-------------------------------------------------------------------------------------
void PipManager::finalizeTaskUninstall(const QString& error, const QString& output)
{
    if (error != "" && output == "")
    {
        emit pipRequestFinished(taskUninstall, "Error uninstalling package\n", false);
    }
    else
    {
        listAvailablePackages();
    }
}

//-------------------------------------------------------------------------------------
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
    if (m_currentTask == taskFetchPackagesDetails)
    {
        m_fetchDetailCancelRequested = true;
    }

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
