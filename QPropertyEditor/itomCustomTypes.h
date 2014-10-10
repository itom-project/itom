/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2014, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.
   
    In addition, as a special exception, the Institut für Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef ITOMCUSTOMTYPES_H
#define ITOMCUSTOMTYPES_H

#include <qvariant.h>
#include "Property.h"

class Property;
class QObject;

//struct Vec3f
//{
//	Vec3f() : X(0.0f), Y(0.0f), Z(0.0f) {} 
//	Vec3f(float x, float y, float z) : X(x), Y(y), Z(z) {}
//	float X, Y, Z;
//
//	bool operator == (const Vec3f& other) const {return X == other.X && Y == other.Y && Z == other.Z;} 
//	bool operator != (const Vec3f& other) const {return X != other.X || Y != other.Y || Z != other.Z;} 
//
//};
//Q_DECLARE_METATYPE(Vec3f)
//
//struct CoordSys
//{
//    CoordSys() : m_x(0.0f), m_y(0.0f), m_z(0.0f), m_scale(1.0f), m_visible(true) {}
//    CoordSys(float x, float y, float z, float scale, bool visible) : m_x(x), m_y(y), m_z(z), m_scale(scale), m_visible(visible) {}
//    float m_x, m_y, m_z, m_scale;
//    bool m_visible;
//
//    bool operator == (const CoordSys& other) const {return m_x == other.m_x && m_y == other.m_y && m_z == other.m_z && m_scale == other.m_scale && m_visible == other.m_visible;} 
//	bool operator != (const CoordSys& other) const {return m_x != other.m_x && m_y != other.m_y && m_z != other.m_z && m_scale != other.m_scale && m_visible != other.m_visible;} 
//};
//Q_DECLARE_METATYPE(CoordSys)



namespace ito
{
    namespace itomCustomTypes
    {
	    void registerTypes();
	    Property* createCustomProperty(const QString& name, QObject* propertyObject, Property* parent);
    }


    class AutoIntervalProperty : public Property
    {
        Q_OBJECT
        Q_PROPERTY(float minimum READ minimum WRITE setMinimum DESIGNABLE true USER true)
        Q_PROPERTY(float maximum READ maximum WRITE setMaximum DESIGNABLE true USER true)
        Q_PROPERTY(bool autoScaling READ autoScaling WRITE setAutoScaling DESIGNABLE true USER true)

    public:
        AutoIntervalProperty(const QString& name = QString(), QObject* propertyObject = 0, QObject* parent = 0);

        QVariant value(int role = Qt::UserRole) const;
        virtual void setValue(const QVariant& value);

        void setEditorHints(const QString& hints);

        float minimum() const;
        void setMinimum(float minimum);

        float maximum() const;
        void setMaximum(float maximum);

        bool autoScaling() const;
        void setAutoScaling(bool autoScaling);

    private:
        QString parseHints(const QString& hints, const QChar component);

        Property*	m_minimum;
        Property*	m_maximum;
        Property*	m_autoScaling;
    };

}
#endif