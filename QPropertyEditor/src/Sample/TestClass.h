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

#ifndef TESTCLASS_H_
#define TESTCLASS_H_

#include <Qt/qobject.h>
#include <Qt/qcolor.h>

#include "CustomTypes.h"

class TestClass : public QObject
{
    Q_OBJECT

    Q_PROPERTY(QString Name READ name WRITE setName DESIGNABLE true USER true)
    Q_PROPERTY(Vec3f Position READ position WRITE setPosition DESIGNABLE true USER true)
    Q_CLASSINFO("Position", "minimumX=-2147483647;maximumX=2147483647;minimumY=-2147483647;maximumY=2147483647;minimumZ=-2147483647;maximumZ=2147483647;");
    Q_PROPERTY(float Radius READ radius WRITE setRadius DESIGNABLE true USER true)
    Q_CLASSINFO("Radius", "minimum=0;maximum=400;decimals=4;singleStep=0.001;");
    Q_PROPERTY(QColor Color READ color WRITE setColor DESIGNABLE true USER true)
    Q_PROPERTY(int SomeNumber READ someNumber DESIGNABLE true USER true)
    Q_PROPERTY(Simpson SimpsonsCharacter READ simpson WRITE setSimpson DESIGNABLE true USER true)
    Q_ENUMS(Simpson)

public:
    enum Simpson { HOMER, MARGE, BART, LISA, MAGGIE };

    TestClass(QObject* parent = 0);
    virtual ~TestClass();


    QString name() const {return m_name;}
    void setName(const QString& name);

    Vec3f position() const {return m_position;}
    void setPosition(const Vec3f& position);

    float radius() const {return m_radius;}
    void setRadius(float radius);

    QColor color() const {return m_color;}
    void setColor(const QColor& color);

    int someNumber() const {return m_someNumber;}

    Simpson simpson() const { return m_simpson; }
    void setSimpson(const Simpson s);
private:

    QString        m_name;
    Vec3f        m_position;
    float        m_radius;
    QColor        m_color;
    const int    m_someNumber;
    Simpson        m_simpson;
};
#endif
