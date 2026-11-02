#define ITOM_IMPORT_API
#include "apiFunctionsInc.h"
#undef ITOM_IMPORT_API
#include "../AddInManager/addInManager.h"

#if WIN32
#include <Windows.h>
//#include <Setupapi.h>
#endif
#include "gtest/gtest.h"
#include <memory>
#include <qcoreapplication.h>
#include <qfileinfo.h>
#if (QT_VERSION >= QT_VERSION_CHECK(5, 9, 0))
#include <qoperatingsystemversion.h>
#endif

//----------------------------------------------------------------------------------------------------------------------------------
/*!
 *   Auxiliary function converting QVector<ito::Param> to QVector<ito::ParamBase>
 *
 *   \param *vecIn    Vector with input parameters
 *   \param &vecOut    Vector with output parameters
 *   \return     0
 */
void convertQVP2QVPB(const QVector<ito::Param>* vecIn, QVector<ito::ParamBase>& vecOut)
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

    // if we want to use multithreading and synchronous plugin calls we need sort of a qt main
    // function this can be either an instance of QApplication (normally used for gui-based
    // applications) or an instance of QCoreApplication, which is reported to be sort of
    // lightweighter. So that is what we / why we do this here Actually now AddInManager should do
    // this for us
    int argc = 0;
    QCoreApplication a(argc, nullptr);

    ito::AddInManager* addInMgrInst = ito::AddInManager::createInstance("", nullptr, nullptr, nullptr);
    EXPECT_NE(addInMgrInst == nullptr, 1);

    retval += addInMgrInst->setTimeOuts(30000, 5000);
    ito::ITOM_API_FUNCS = addInMgrInst->getItomApiFuncsPtr();
    QString addInPath("");
#if WIN32
#ifdef _DEBUG
    QString aimName("addinmanagerd.dll");
#else
    QString aimName("addinmanager.dll");
#endif

#ifdef UNICODE
    wchar_t path[_MAX_PATH + 1];
    wchar_t aimName2[_MAX_PATH + 1];
    int len = aimName.toWCharArray(aimName2);
    aimName2[len] = 0;
#else
    char aimName2[_MAX_PATH + 1];
    char path[_MAX_PATH + 1];
#endif

    GetModuleFileName(GetModuleHandle(aimName2), path, sizeof(path) / sizeof(path[0]));

#ifdef UNICODE
    QString path2 = QString::fromWCharArray(path);
#else
    QString path2 = QString::fromUtf8(path);
#endif

    QFileInfo fi = QFileInfo(path2);

    if (fi.exists())
    {
        addInPath = fi.absolutePath();
    }
    else
    {
        GetModuleFileName(GetModuleHandle(aimName2), path, sizeof(path) / sizeof(path[0]));
#ifdef UNICODE
        path2 = QString::fromWCharArray(path);
#else
        path2 = QString::fromUtf8(path);
#endif
        fi = QFileInfo(path2);

        if (fi.exists())
        {
            addInPath = fi.absolutePath();
        }
        else
        {
            addInPath = "";
        }
    }

    char* oldpath = getenv("path");

    // try add or lib directory to path variables, to avoid user has to do this
    QString libDir = addInPath + QString("/lib");
    char* newpath = (char*)malloc(strlen(oldpath) + libDir.size() + 7);
    newpath[0] = 0;
    strcat(newpath, "path=");

#if WINVER > 0x0502
#if (QT_VERSION >= QT_VERSION_CHECK(5, 9, 0))
    if (QOperatingSystemVersion::current() >= QOperatingSystemVersion::Windows7)
#else
    if (QSysInfo::windowsVersion() > QSysInfo::WV_XP)
#endif
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
    QVector<ito::Param>*paramsMand = NULL, *paramsOpt = NULL;
    QVector<ito::ParamBase> paramsMandCpy, paramsOptCpy;

    retval = addInMgrInst->getInitParams(
        "DummyMotor", ito::typeActuator, &pluginNum, paramsMand, paramsOpt);
    if (retval.containsWarningOrError())
    {
        // The plugin DummyMotor is part of the separate repository 'plugins'. A build of the core
        // repository alone does not contain it. Without the plugin, this test cannot be executed.
        addInMgrInst->closeInstance();
        GTEST_SKIP() << "The plugin DummyMotor was not found in " << addInPath.toStdString()
                     << "/plugins: " << retval.errorMessage();
    }

    int enableAutoLoadParams = 1;
    ito::AddInActuator* dummyMot = nullptr;
    convertQVP2QVPB(paramsMand, paramsMandCpy);
    convertQVP2QVPB(paramsOpt, paramsOptCpy);
    // this could be an alternative call, in case we run single threaded
    //        if (QMetaObject::invokeMethod(addInMgrInst, "initAddIn", Qt::DirectConnection,
    //        Q_ARG(int, pluginNum), Q_ARG(QString, auxPluginName), Q_ARG(ito::AddInDataIO**,
    //        &auxCtrl[listID]), Q_ARG(QVector<ito::Param>*, paramsMand),
    //        Q_ARG(QVector<ito::Param>*, paramsOpt), Q_ARG(bool, enableAutoLoadParams)))
    retval += addInMgrInst->initAddIn(
        pluginNum,
        "DummyMotor",
        &dummyMot,
        &paramsMandCpy,
        &paramsOptCpy,
        enableAutoLoadParams,
        nullptr);

    EXPECT_NE(dummyMot == NULL, 1);
    EXPECT_EQ(retval, ito::retOk);

    retval += addInMgrInst->closeAddIn(dummyMot, nullptr);
    EXPECT_EQ(retval, ito::retOk);

    addInMgrInst->closeInstance();
};

//----------------------------------------------------------------------------------------------------------------------------------
// The following tests do not require any plugin.

namespace
{
//! creates the AddInManager without scanning for plugins and closes it at the end of the test.
class AddInManagerWithoutPlugins : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_app.reset(new QCoreApplication(m_argc, nullptr));
        m_manager = ito::AddInManager::createInstance("", nullptr, nullptr, nullptr);
        ASSERT_NE(m_manager, nullptr);
        ito::ITOM_API_FUNCS = m_manager->getItomApiFuncsPtr();
    }

    void TearDown() override
    {
        ito::AddInManager::closeInstance();
        m_app.reset();
    }

    int m_argc = 0;
    std::unique_ptr<QCoreApplication> m_app;
    ito::AddInManager* m_manager = nullptr;
};
} // namespace

// Regression test: an invalid plugin index crashed the manager (out-of-range access in a list),
// instead of returning an error.
TEST_F(AddInManagerWithoutPlugins, InitActuatorWithInvalidPluginIndexReturnsError)
{
    QVector<ito::ParamBase> mandatory, optional;

    for (int index : {-1, 0, 1000})
    {
        ito::AddInActuator* actuator = nullptr;
        ito::RetVal retval = m_manager->initAddIn(
            index, "NotExistingPlugin", &actuator, &mandatory, &optional, false, nullptr);

        EXPECT_TRUE(retval.containsError()) << "plugin index " << index;
        EXPECT_EQ(actuator, nullptr) << "plugin index " << index;
    }
}

TEST_F(AddInManagerWithoutPlugins, InitDataIOWithInvalidPluginIndexReturnsError)
{
    QVector<ito::ParamBase> mandatory, optional;

    for (int index : {-1, 0, 1000})
    {
        ito::AddInDataIO* dataIO = nullptr;
        ito::RetVal retval = m_manager->initAddIn(
            index, "NotExistingPlugin", &dataIO, &mandatory, &optional, false, nullptr);

        EXPECT_TRUE(retval.containsError()) << "plugin index " << index;
        EXPECT_EQ(dataIO, nullptr) << "plugin index " << index;
    }
}

TEST_F(AddInManagerWithoutPlugins, InitAlgoWithInvalidPluginIndexReturnsError)
{
    QVector<ito::ParamBase> mandatory, optional;

    for (int index : {-1, 0, 1000})
    {
        ito::AddInAlgo* algo = nullptr;
        ito::RetVal retval = m_manager->initAddIn(
            index, "NotExistingPlugin", &algo, &mandatory, &optional, false, nullptr);

        EXPECT_TRUE(retval.containsError()) << "plugin index " << index;
        EXPECT_EQ(algo, nullptr) << "plugin index " << index;
    }
}
