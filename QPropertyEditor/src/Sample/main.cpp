// *************************************************************************************************
//
// This code is part of the Sample Application to demonstrate the use of the QPropertyEditor library.
// It is distributed as public domain and can be modified and used by you without any limitations.
//
// Your feedback is welcome!
//
// Author: Volker Wiendl
// Enum enhancement by Roman alias banal from qt-apps.org
// *************************************************************************************************

#include <Qt/qapplication.h>
#include "SampleApp.h"

int main(int argc, char** argv)
{
    QApplication app( argc, argv );

    // create a new instance of SampleApp
    SampleApp mainWindow;
    mainWindow.show();

    // Enters the main event loop and waits until exit() is called
    // or the main widget is destroyed, and Returns the value that
    // was set via to exit() (which is 0 if exit() is called via quit()).
    return app.exec();
}
