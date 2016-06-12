/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#include "qpropertyHelper.h"

#include <qvector2d.h>
#include <qvector3d.h>
#include <qvector4d.h>
#include <qrect.h>



namespace ito {

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ QVariant QPropertyHelper::QVariantCast(const QVariant &item, int userDestType, ito::RetVal &retval)
{
    if (item.userType() == userDestType)
    {
        retval += ito::retOk;
        return item;
    }

    bool ok = false;
    QVariant result;

    if (item.type() == QVariant::List)
    {
        if (userDestType == QVariant::PointF)
        {
            const QVariantList list = item.toList();
            if (list.size() == 2)
            {
                bool ok2;
                result = QPointF(list[0].toFloat(&ok), list[1].toFloat(&ok2));
                ok &= ok2;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, "transformation error to PointF: at least one value could not be transformed to float.");
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, "transformation error to PointF: 2 values required.");
            }
        }
        else if (userDestType == QVariant::Point)
        {
            const QVariantList list = item.toList();
            if (list.size() == 2)
            {
                bool ok2;
                result = QPoint(list[0].toInt(&ok), list[1].toInt(&ok2));
                ok &= ok2;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, "transformation error to Point: at least one value could not be transformed to integer.");
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, "transformation error to Point: 2 values required.");
            }
        }
        else if (userDestType == QVariant::Rect)
        {
            const QVariantList list = item.toList();
            if (list.size() == 4)
            {
                bool ok2, ok3, ok4;
                result = QRect(list[0].toInt(&ok), list[1].toInt(&ok2), list[2].toInt(&ok3), list[3].toInt(&ok4));
                ok &= ok2;
                ok &= ok3;
                ok &= ok4;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, "transformation error to Rect: at least one value could not be transformed to integer.");
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, "transformation error to Rect: 4 values required.");
            }
        }
        else if (userDestType == QVariant::RectF)
        {
            const QVariantList list = item.toList();
            if (list.size() == 4)
            {
                bool ok2, ok3, ok4;
                result = QRectF(list[0].toFloat(&ok), list[1].toFloat(&ok2), list[2].toFloat(&ok3), list[3].toFloat(&ok4));
                ok &= ok2;
                ok &= ok3;
                ok &= ok4;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, "transformation error to RectF: at least one value could not be transformed to float.");
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, "transformation error to RectF: 4 values required.");
            }
        }
        else if (userDestType == QVariant::Vector2D)
        {
            const QVariantList list = item.toList();
            if (list.size() == 2)
            {
                bool ok2;
                result = QVector2D(list[0].toFloat(&ok), list[1].toFloat(&ok2));
                ok &= ok2;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, "transformation error to Vector2D: at least one value could not be transformed to float.");
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, "transformation error to Vector2D: 2 values required.");
            }
        }
        else if (userDestType == QVariant::Vector3D)
        {
            const QVariantList list = item.toList();
            if (list.size() == 3)
            {
                bool ok2, ok3;
                result = QVector3D(list[0].toFloat(&ok), list[1].toFloat(&ok2), list[2].toFloat(&ok3));
                ok &= ok2;
                ok &= ok3;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, "transformation error to Vector3D: at least one value could not be transformed to float.");
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, "transformation error to Vector3D: 3 values required.");
            }
        }
        else if (userDestType == QVariant::Vector4D)
        {
            const QVariantList list = item.toList();
            if (list.size() == 4)
            {
                bool ok2, ok3, ok4;
                result = QVector4D(list[0].toFloat(&ok), list[1].toFloat(&ok2), list[2].toFloat(&ok3), list[3].toFloat(&ok4));
                ok &= ok2;
                ok &= ok3;
                ok &= ok4;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, "transformation error to Vector4D: at least one value could not be transformed to float.");
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, "transformation error to Vector4D: 4 values required.");
            }
        }
        else if (userDestType == QVariant::Size)
        {
            const QVariantList list = item.toList();
            if (list.size() == 2)
            {
                bool ok2;
                result = QSize(list[0].toInt(&ok), list[1].toInt(&ok2));
                ok &= ok2;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, "transformation error to Size: at least one value could not be transformed to integer.");
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, "transformation error to Size: 2 values required.");
            }
        }
        else if (userDestType == QMetaType::type("ito::AutoInterval"))
        {
            const QVariantList list = item.toList();
            if (list.size() == 2)
            {
                bool ok2;
                result = QVariant::fromValue<ito::AutoInterval>(ito::AutoInterval(list[0].toFloat(&ok), list[1].toFloat(&ok2)));
                ok &= ok2;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, "transformation error to AutoInterval: at least one value could not be transformed to float.");
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, "transformation error to AutoInterval: 2 values required.");
            }
        }
        else if (userDestType == QMetaType::type("QVector<ito::Shape>"))
        {
            const QVariantList list = item.toList();
            QVector<ito::Shape> shapes;

            if (list.size() > 0)
            {
                foreach(const QVariant &listItem, list)
                {
                    if (listItem.type() == QVariant::UserType && listItem.userType() == QMetaType::type("ito::Shape"))
                    {
                        shapes.append(qvariant_cast<ito::Shape>(listItem));
                        ok = true;
                    }
                    else
                    {
                        retval += ito::RetVal(ito::retError, 0, "transformation error to vector of shapes: at least one item could not be interpreted as shape.");
                        ok = false;
                        break;
                    }
                }
            }
            else
            {
                ok = true;
            }

            if (!retval.containsError())
            {
                result = QVariant::fromValue<QVector<ito::Shape> >(shapes);
            }
        }
    } //end item.type() == QVariant::List
    else if (item.type() == QVariant::String)
    {
        if (userDestType == QMetaType::type("ito::AutoInterval"))
        {
            const QString str = item.toString();
            if (QString::compare(str, "auto", Qt::CaseInsensitive) == 0 || QString::compare(str, "<auto>", Qt::CaseInsensitive) == 0)
            {
                ito::AutoInterval ival;
                ival.setAuto(true);
                result = QVariant::fromValue<ito::AutoInterval>(ival);
                ok = true;
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, "transformation error to AutoInterval: value must be [min,max] or 'auto'.");
            }
        }
    } //end item.type() == QVariant::String
    else if (userDestType == QVariant::Color)
    {
        bool ok2;
        uint value = item.toUInt(&ok2);
        if (ok2)
        {
            result = QColor((value & 0xff0000) >> 16, (value & 0x00ff00) >> 8, value & 0x0000ff);
            ok = true;
        }
    }
    else if (item.userType() == QMetaType::type("ito::Shape"))
    {
        if (userDestType == QMetaType::type("QVector<ito::Shape>"))
        {
            QVector<ito::Shape> shapes;
            shapes << qvariant_cast<ito::Shape>(item);
            ok = true;
            result = QVariant::fromValue<QVector<ito::Shape> >(shapes);
        }
    }
    else if (item.userType() == QMetaType::type("ito::PythonNone"))
    {
        if (userDestType == QMetaType::type("QSharedPointer<ito::DataObject>"))
        {
            ok = true;
            result = QVariant::fromValue<QSharedPointer<ito::DataObject> >(QSharedPointer<ito::DataObject>(new ito::DataObject()));
        }
        else if (userDestType == QMetaType::type("QPointer<ito::AddInDataIO>"))
        {
            ok = true;
            result = QVariant::fromValue<QPointer<ito::AddInDataIO> >(QPointer<ito::AddInDataIO>());
        }
        else if (userDestType == QMetaType::type("QPointer<ito::AddInActuator>"))
        {
            ok = true;
            result = QVariant::fromValue<QPointer<ito::AddInActuator> >(QPointer<ito::AddInActuator>());
        }
        else if (userDestType == QMetaType::type("QVector<ito::Shape>"))
        {
            ok = true;
            result = QVariant::fromValue<QVector<ito::Shape> >(QVector<ito::Shape>());
        }
#if ITOM_POINTCLOUDLIBRARY > 0   
        else if (userDestType == QMetaType::type("QSharedPointer<ito::PCLPointCloud>"))
        {
            ok = true;
            result = QVariant::fromValue<QSharedPointer<ito::PCLPointCloud> >(QSharedPointer<ito::PCLPointCloud>(new ito::PCLPointCloud()));
        }
        else if (userDestType == QMetaType::type("QSharedPointer<ito::PCLPolygonMesh>"))
        {
            ok = true;
            result = QVariant::fromValue<QSharedPointer<ito::PCLPolygonMesh> >(QSharedPointer<ito::PCLPolygonMesh>(new ito::PCLPolygonMesh()));
        }
#endif //#if ITOM_POINTCLOUDLIBRARY > 0
    }


    if (!ok && !retval.containsError()) //not yet converted, try to convert it using QVariant internal conversion method
    {
#if QT_VERSION < 0x050000
        if (userDestType < QVariant::UserType && item.canConvert((QVariant::Type)userDestType))
#else
        if (item.canConvert(userDestType))
#endif
        {
            result = item;
#if QT_VERSION < 0x050000
            result.convert((QVariant::Type)userDestType);
#else
            result.convert(userDestType);
#endif
            ok = true;
        }
        else
        {
            QString fromName, toName;
            if (QMetaType::isRegistered(item.userType()))
            {
                fromName = QMetaType::typeName(item.userType());
            }
            else
            {
                fromName = "unknown";
            }

            if (QMetaType::isRegistered(userDestType))
            {
                toName = QMetaType::typeName(userDestType);
            }
            else
            {
                toName = "unknown";
            }

            retval += ito::RetVal::format(ito::retError, 0, "no conversion from QVariant type %s to %s is possible", fromName.toLatin1().data(), toName.toLatin1().data());
        }
    }

    if (ok)
    {
        return result;
    }
    return item;
}

//----------------------------------------------------------------------------------------------------------------------------------
/*static*/ QVariant QPropertyHelper::QVariantToEnumCast(const QVariant &item, const QMetaEnum &enumerator, ito::RetVal &retval)
{
    int val;
    bool ok;
    val = item.toInt(&ok);
    QVariant result;

    if (ok) //integer
    {
        if (enumerator.isFlag())
        {
            int result_ = 0;
            int e;

            for (int idx = 0; idx < enumerator.keyCount(); ++idx)
            {
                e = enumerator.value(idx);
                if (val & e)
                {
                    result_ |= e;
                }
            }

            if (result_ == val)
            {
                result = result_;
            }
            else
            {
                retval += ito::RetVal::format(ito::retError, 0, "The value %i contains a bitmask that is not fully covered by an or-combination of the enumeration %s::%s (flags)", val, enumerator.scope(), enumerator.name());
                return result;
            }
        }
        else
        {
            const char *key = enumerator.valueToKey(val);
            if (key)
            {
                result = val;
            }
            else
            {
                retval += ito::RetVal::format(ito::retError, 0, "The value %i does not exist in the enumeration %s::%s", val, enumerator.scope(), enumerator.name());
                return result;
            }
        }
    }
    else //
    {
        if (item.canConvert(QVariant::String)) //string
        {
            QString str = item.toString();
            if (enumerator.isFlag())
            {
                int result_ = 0;
                QStringList str_ = str.split(";");
                foreach(const QString &substr, str_)
                {
                    if (substr.isEmpty() == false)
                    {
                        val = enumerator.keyToValue(substr.toLatin1().data());
                        if (val >= 0)
                        {
                            result_ |= val;
                        }
                        else
                        {
                            retval += ito::RetVal::format(ito::retError, 0, "The key %s does not exist in the enumeration %s::%s (flags)", str.toLatin1().data(), enumerator.scope(), enumerator.name());
                            return result;
                        }
                    }
                }

                result = result_;
            }
            else
            {
                val = enumerator.keyToValue(str.toLatin1().data());
                if (val >= 0)
                {
                    result = val;
                }
                else
                {
                    retval += ito::RetVal::format(ito::retError, 0, "The key %s does not exist in the enumeration %s::%s", str.toLatin1().data(), enumerator.scope(), enumerator.name());
                    return result;
                }
            }
        }
        else
        {
            retval += ito::RetVal::format(ito::retError, 0, "Use an integer or a string for a value of the enumeration %s::%s", enumerator.scope(), enumerator.name());
            return result;
        }
    }

    return result;
}



//----------------------------------------------------------------------------------------------------------------------------------
RetVal QPropertyHelper::readProperty(const QObject *object, const char* propName, QVariant &value)
{
    RetVal retValue;
    
    if (object)
    {
        value = object->property(propName);
        if (value.isValid())
        {
            retValue += RetVal::format(retError, 0, "property '%s' could not be read", propName);
        }
    }
    else
    {
        retValue += RetVal(retError, 0, "invalid object");
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal QPropertyHelper::writeProperty(QObject *object, const char* propName, const QVariant &value)
{
    RetVal retValue;

    if (object)
    {
        QStringList errString;
        const QMetaObject *mo = object->metaObject();
        QMetaProperty prop;
        int index = mo->indexOfProperty(propName);
        if (index >= 0)
        {
            prop = mo->property(index);

            //check whether types need to be casted
            //e.g. QVariantList can sometimes be casted to QPointF...
            //bool ok;
            RetVal tempRet;
            QVariant item;

            if (prop.isWritable() == false)
            {
                retValue += ito::RetVal::format(ito::retError, 0, "Property '%s' is not writeable.", propName);
            }
            else if (prop.isEnumType())
            {
                item = QPropertyHelper::QVariantToEnumCast(value, prop.enumerator(), retValue);
            }
            else
            {
                item = QPropertyHelper::QVariantCast(value, prop.userType(), retValue);
            }

            if (!prop.write(object, item))
            {
                retValue += ito::RetVal::format(ito::retError, 0, "Property '%s' could not be set. Maybe wrong input type.", propName);
            }

        }
        else
        {
            retValue += ito::RetVal::format(ito::retError, 0, "Property '%s' does not exist.", propName);
        }
    }
    else
    {
        retValue += RetVal(retError, 0, "Invalid object");
    }

    return retValue;
}

} //end namespace ito
