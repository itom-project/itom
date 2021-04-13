#define ITOM_IMPORT_API
#include "apiFunctionsInc.h"
#undef ITOM_IMPORT_API
#include "../AddInManager/addInManager.h"

#if WIN32
#include <Windows.h>
//#include <Setupapi.h>
#endif
#include "gtest/gtest.h"
#include <qcoreapplication.h>
#include <qfileinfo.h>

//----------------------------------------------------------------------------------------------------------------------------------
/*!
 *   Auxiliar function converting QVector<ito::Param> to QVector<ito::ParamBase>
 *
 *   \param *vecIn    Vector with input parameters
 *   \param &vecOut    Vector with output parameters
 *   \return     0
 */
void convertQVP2QVPB(const QVector<ito::Param> *vecIn, QVector<ito::ParamBase> &vecOut)
{
    vecOut.clear();
    if (vecIn)
    {
        vecOut.reserve(vecIn->size());
        for (int np = 0; np < vecIn->size(); np++)
        {
            vecOut.append(vecIn->at(np));
        }
    }
}

TEST(AddInManagerTest, General)
{
    ito::RetVal retval;

    // if we want to use multithreading and synchroneous plugin calls we need sort of a qt main function
    // this can be either an instance of QApplication (normally used for gui-based applications)
    // or an instance of QCoreApplication, which is reported to be sort of lightweighter.
    // So that is what we / why we do this here
    // Actually now AddInManager should do this for us
    int argc = 0;
    QCoreApplication a(argc, NULL);

    ito::AddInManager *addInMgrInst = ito::AddInManager::createInstance("", NULL, NULL, NULL);
    EXPECT_NE(addInMgrInst == NULL, 1);

    retval += addInMgrInst->setTimeOuts(30000, 5000);
    ito::ITOM_API_FUNCS = addInMgrInst->getItomApiFuncsPtr();
    QString addInPath("");
#if WIN32
#ifdef _DEBUG
    QString aimName("addinmanagerd.dll");
#else
    QString aimName("addinmanager.dll");
#endif
    char path[_MAX_PATH + 1];
    GetModuleFileName(GetModuleHandle(aimName.toLatin1().data()), path, sizeof(path) / sizeof(path[0]));
    QFileInfo fi = QFileInfo(path);
    if (fi.exists())
        addInPath = fi.absolutePath();
    else
    {
        GetModuleFileName(GetModuleHandle(aimName.toLatin1().data()), path, sizeof(path) / sizeof(path[0]));
        fi = QFileInfo(path);
        if (fi.exists())
            addInPath = fi.absolutePath();
        else
            addInPath = "";
    }

    char *oldpath = getenv("path");
    char pathSep[] = ";";

    // try add or lib directory to path variables, to avoid user has to do this
    QString libDir = addInPath + QString("/lib");
    char *newpath = (char *)malloc(strlen(oldpath) + libDir.size() + 7);
    newpath[0] = 0;
    strcat(newpath, "path=");

#if WINVER > 0x0502
    if (QSysInfo::windowsVersion() > QSysInfo::WV_XP)
    {
        SetDllDirectoryA(libDir.toLatin1().data());
    }
#endif
    strcat(newpath, libDir.toLatin1().data()); // set libDir at the beginning of the path-variable
    strcat(newpath, ";");
    strcat(newpath, oldpath);
    _putenv(newpath);
#endif
    retval += addInMgrInst->scanAddInDir(addInPath + QString("/plugins"));
    // if (retval.containsError())
    //    goto EXIT;

    // initializing dummy motor
    int pluginNum;
    QVector<ito::Param> *paramsMand = NULL, *paramsOpt = NULL;
    QVector<ito::ParamBase> paramsMandCpy, paramsOptCpy;

    retval = addInMgrInst->getInitParams("DummyMotor", ito::typeActuator, &pluginNum, paramsMand, paramsOpt);
    if (retval.containsWarningOrError())
    {
        QString errMsg = QString("Error loading Dummy-Motor: ") + QString::fromLatin1(retval.errorMessage());
        retval = ito::RetVal(ito::retError, 0, errMsg.toLatin1().data());
        //        goto EXIT;
    }

    int enableAutoLoadParams = 1;
    ito::AddInActuator *dummyMot;
    convertQVP2QVPB(paramsMand, paramsMandCpy);
    convertQVP2QVPB(paramsOpt, paramsOptCpy);
    // this could be an alternative call, in case we run single threaded
    //        if (QMetaObject::invokeMethod(addInMgrInst, "initAddIn", Qt::DirectConnection, Q_ARG(int, pluginNum),
    //        Q_ARG(QString, auxPluginName), Q_ARG(ito::AddInDataIO**, &auxCtrl[listID]), Q_ARG(QVector<ito::Param>*,
    //        paramsMand), Q_ARG(QVector<ito::Param>*, paramsOpt), Q_ARG(bool, enableAutoLoadParams)))
    retval += addInMgrInst->initAddIn(pluginNum, "DummyMotor", &dummyMot, &paramsMandCpy, &paramsOptCpy,
                                      enableAutoLoadParams, NULL);
    EXPECT_NE(dummyMot == NULL, 1);

    addInMgrInst->closeInstance();
};
