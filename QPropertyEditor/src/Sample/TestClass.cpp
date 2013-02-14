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

#include "TestClass.h"


TestClass::TestClass(QObject* parent /*= 0*/) : QObject(parent), 
m_name("Test Class"), m_position(10, 10, 10), m_radius(15.0f), m_color(Qt::red), m_someNumber(14),
m_simpson(HOMER)
{
	
}


TestClass::~TestClass()
{
}

void TestClass::setName(const QString &name)
{
	m_name = name;
}

void TestClass::setPosition(const Vec3f &position)
{
	m_position = position;
}

void TestClass::setRadius(float radius)
{
	m_radius = radius;
}

void TestClass::setColor(const QColor &color)
{
	m_color = color;
}

void TestClass::setSimpson(const Simpson s)
{
	m_simpson = s;
}