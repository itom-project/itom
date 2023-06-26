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

#ifndef TESTSUBCLASS_H_
#define TESTSUBCLASS_H_

#include "TestClass.h"

class TestSubClass : public TestClass
{
    Q_OBJECT
    Q_CLASSINFO("TestSubClass", "Sub Class Specific");

    Q_PROPERTY(QString Date READ date WRITE setDate DESIGNABLE true USER true)
    Q_CLASSINFO("Date", "inputMask=0000-00-00;");

public:
    TestSubClass(QObject* parent = 0);
    virtual ~TestSubClass();

    QString date() const {return m_date;}
    void setDate(const QString& date) {m_date = date;}

private:
    QString m_date;

};
#endif
