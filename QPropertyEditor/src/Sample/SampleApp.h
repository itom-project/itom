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

#ifndef SAMPLEAPP_H_
#define SAMPLEAPP_H_

#include "./ui_SampleApp.h"

class TestClass;
class TestSubClass;

/**
 * This Sample Application demonstrates the use of the QPropertyEditor Library
 * It contains only a dock widget with a QPropertyEditor widget in it.
 * In this property editor the properties of two classes (TestClass and TestSubClass)
 * are displayed for demonstration purposes
 */
class SampleApp : public QMainWindow, protected Ui_SampleApp
{
    Q_OBJECT
public:
    /**
     * Constructor
     * @param parent optional parent widget this window will be child of
     * @param flags optional window flags
     */
    SampleApp(QWidget* parent = 0, Qt::WFlags flags = 0);

    /// Destructor
    virtual ~SampleApp();

private:
    /// Testclass with some properties
    TestClass*        m_testClass;
    /// Another test class inherited from TestClass to show the hierarchy managment of the QPropertyEditor
    TestSubClass*    m_testSubClass;
};
#endif
