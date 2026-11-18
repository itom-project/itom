#include "gtest/gtest.h"

#include "codeEditorTestHelper.h"

#include <qapplication.h>
#include <qdir.h>
#include <qtemporarydir.h>

#include <cstring>
#include <iostream>

int main(int argc, char* argv[])
{
    // the code editor is a widget and needs a QApplication. The platform plugin
    // "offscreen" allows to run the test without a display (e.g. in CI).
    if (!qEnvironmentVariableIsSet("QT_QPA_PLATFORM"))
    {
        qputenv("QT_QPA_PLATFORM", "offscreen");
    }

    QApplication app(argc, argv);

    // every test gets a fresh settings file in a temporary directory
    QTemporaryDir tempDir;
    ito::test::setSettingsFilePath(tempDir.path() + QDir::separator() + "itom.ini");

    ::testing::InitGoogleTest(&argc, argv); // Initializing the google test.

    int result = RUN_ALL_TESTS(); // To start Test check

    bool executedByGoogleTestAdapter = false;

    for (int i = 0; i < argc; ++i)
    {
        const char* arg = argv[i];

        if (strcmp(arg, "-googletestadapter") == 0)
        {
            executedByGoogleTestAdapter = true;
            break;
        }
    }

    if (!executedByGoogleTestAdapter)
    {
        // execute this only, if the unittest is directly executed.
        // If it is executed by the VS Google Test Adapter extension,
        // no keyboard interaction must be implemented.
        std::system("pause");
    }

    return result;
}
