/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
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

Q_DECLARE_METATYPE(QVector<int>)


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
        const QVariantList list = item.toList();

        if (userDestType == QVariant::PointF)
        {
            if (list.size() == 2)
            {
                bool ok2;
                result = QPointF(list[0].toFloat(&ok), list[1].toFloat(&ok2));
                ok &= ok2;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to PointF: at least one value could not be transformed to float.").toLatin1().data());
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to PointF: 2 values required.").toLatin1().data());
            }
        }
        else if (userDestType == QVariant::Point)
        {
            if (list.size() == 2)
            {
                bool ok2;
                result = QPoint(list[0].toInt(&ok), list[1].toInt(&ok2));
                ok &= ok2;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to Point: at least one value could not be transformed to integer.").toLatin1().data());
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to Point: 2 values required.").toLatin1().data());
            }
        }
        else if (userDestType == QVariant::Rect)
        {
            if (list.size() == 4)
            {
                bool ok2, ok3, ok4;
                result = QRect(list[0].toInt(&ok), list[1].toInt(&ok2), list[2].toInt(&ok3), list[3].toInt(&ok4));
                ok &= ok2;
                ok &= ok3;
                ok &= ok4;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to Rect: at least one value could not be transformed to integer.").toLatin1().data());
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to Rect: 4 values required.").toLatin1().data());
            }
        }
        else if (userDestType == QVariant::RectF)
        {
            if (list.size() == 4)
            {
                bool ok2, ok3, ok4;
                result = QRectF(list[0].toFloat(&ok), list[1].toFloat(&ok2), list[2].toFloat(&ok3), list[3].toFloat(&ok4));
                ok &= ok2;
                ok &= ok3;
                ok &= ok4;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to RectF: at least one value could not be transformed to float.").toLatin1().data());
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to RectF: 4 values required.").toLatin1().data());
            }
        }
        else if (userDestType == QVariant::Vector2D)
        {
            if (list.size() == 2)
            {
                bool ok2;
                result = QVector2D(list[0].toFloat(&ok), list[1].toFloat(&ok2));
                ok &= ok2;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to Vector2D: at least one value could not be transformed to float.").toLatin1().data());
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to Vector2D: 2 values required.").toLatin1().data());
            }
        }
        else if (userDestType == QVariant::Vector3D)
        {
            if (list.size() == 3)
            {
                bool ok2, ok3;
                result = QVector3D(list[0].toFloat(&ok), list[1].toFloat(&ok2), list[2].toFloat(&ok3));
                ok &= ok2;
                ok &= ok3;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to Vector3D: at least one value could not be transformed to float.").toLatin1().data());
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to Vector3D: 3 values required.").toLatin1().data());
            }
        }
        else if (userDestType == QVariant::Vector4D)
        {
            if (list.size() == 4)
            {
                bool ok2, ok3, ok4;
                result = QVector4D(list[0].toFloat(&ok), list[1].toFloat(&ok2), list[2].toFloat(&ok3), list[3].toFloat(&ok4));
                ok &= ok2;
                ok &= ok3;
                ok &= ok4;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to Vector4D: at least one value could not be transformed to float.").toLatin1().data());
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to Vector4D: 4 values required.").toLatin1().data());
            }
        }
        else if (userDestType == QVariant::Size)
        {
            if (list.size() == 2)
            {
                bool ok2;
                result = QSize(list[0].toInt(&ok), list[1].toInt(&ok2));
                ok &= ok2;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to Size: at least one value could not be transformed to integer.").toLatin1().data());
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to Size: 2 values required.").toLatin1().data());
            }
        }
        else if (userDestType == QMetaType::type("ito::AutoInterval"))
        {
            if (list.size() == 2)
            {
                bool ok2;
                result = QVariant::fromValue<ito::AutoInterval>(ito::AutoInterval(list[0].toDouble(&ok), list[1].toDouble(&ok2)));
                ok &= ok2;

                if (!ok)
                {
                    retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to AutoInterval: at least one value could not be transformed to float.").toLatin1().data());
                }
            }
            else
            {
                retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to AutoInterval: 2 values required.").toLatin1().data());
            }
        }
        else if (userDestType == QMetaType::type("QVector<ito::Shape>"))
        {
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
                        retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to vector of shapes: at least one item could not be interpreted as shape.").toLatin1().data());
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
        else if (userDestType == QMetaType::type("QVector<int>"))
        {
            QVector<int> values;
            bool ok_;
            ok = true;

            if (list.size() > 0)
            {
                foreach(const QVariant &listItem, list)
                {
                    values << listItem.toInt(&ok_);

                    if (!ok_)
                    {
                        retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to QVector<int>: at least one value could not be transformed to int.").toLatin1().data());
                        ok = false;
                        break;
                    }
                }
            }

            if (!retval.containsError())
            {
                result = QVariant::fromValue<QVector<int> >(values);
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
                retval += ito::RetVal(ito::retError, 0, QObject::tr("Transformation error to AutoInterval: value must be [min,max] or 'auto'.").toLatin1().data());
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
    else if (item.userType() == QMetaType::type("QPointer<ito::AddInActuator>"))
    {
        if (userDestType == QMetaType::type("QPointer<ito::AddInBase>"))
        {
            QPointer<ito::AddInActuator> actuator = qvariant_cast<QPointer<ito::AddInActuator> >(item);
            QPointer<ito::AddInBase> base = actuator.data();
            ok = true;
            result = QVariant::fromValue<QPointer<ito::AddInBase> >(base);
        }
    }
    else if (item.userType() == QMetaType::type("QPointer<ito::AddInDataIO>"))
    {
        if (userDestType == QMetaType::type("QPointer<ito::AddInBase>"))
        {
            QPointer<ito::AddInDataIO> dataIO = qvariant_cast<QPointer<ito::AddInDataIO> >(item);
            QPointer<ito::AddInBase> base = dataIO.data();
            ok = true;
            result = QVariant::fromValue<QPointer<ito::AddInBase> >(base);
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
        else if (userDestType == QMetaType::type("QPointer<ito::AddInBase>"))
        {
            ok = true;
            result = QVariant::fromValue<QPointer<ito::AddInBase> >(QPointer<ito::AddInBase>());
        }
        else if (userDestType == QMetaType::type("QVector<ito::Shape>"))
        {
            ok = true;
            result = QVariant::fromValue<QVector<ito::Shape> >(QVector<ito::Shape>());
        }
        else if (userDestType == QMetaType::type("ito::ItomPlotHandle"))
        {
            ok = true;
            result = QVariant::fromValue<ito::ItomPlotHandle>(ito::ItomPlotHandle());
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
        if (item.canConvert(userDestType))
        {
            result = item;
            result.convert(userDestType);
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
                fromName = QObject::tr("Unknown");
            }

            if (QMetaType::isRegistered(userDestType))
            {
                toName = QMetaType::typeName(userDestType);
            }
            else
            {
                toName = QObject::tr("Unknown");
            }

            retval += ito::RetVal::format(ito::retError, 0, QObject::tr("No conversion from QVariant type '%s' to '%s' is possible").toLatin1().data(),
                fromName.toLatin1().data(), toName.toLatin1().data());
        }
    }

    if (ok)
    {
        return result;
    }
    return item;
}

//----------------------------------------------------------------------------------------------------------------------------------
//!< returns a help string that contains all possible keys of the enumerator in the form key1 (value1), key2 (value2) ...
QString enumValuesText(const QMetaEnum &enumerator)
{
    QStringList output;

    for (int i = 0; i < enumerator.keyCount(); ++i)
    {
        const char* key = enumerator.key(i);
        output << QString("'%1' (%2)").arg(key).arg(enumerator.keyToValue(key));
    }

    return output.join(", ");
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
                retval += ito::RetVal::format(ito::retError, 0, QObject::tr("The value %i contains a bitmask that is not fully covered " \
                    "by an or-combination of the flags enumeration %s::%s. " \
                    "It can only consist of a combination of the following keys or values: %s.").toLatin1().data(),
                    val, enumerator.scope(), enumerator.name(), enumValuesText(enumerator).toLatin1().data());
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
                retval += ito::RetVal::format(ito::retError, 0, QObject::tr("The value %i does not exist in the enumeration %s::%s. " \
                    "Possible keys or values are: %s.").toLatin1().data(),
                    val, enumerator.scope(), enumerator.name(), enumValuesText(enumerator).toLatin1().data());
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
                            retval += ito::RetVal::format(ito::retError, 0, QObject::tr("The key '%s' does not exist in the flags enumeration %s::%s. " \
                                "It can only consist of a combination of the following keys or values: %s.").toLatin1().data(),
                                str.toLatin1().data(), enumerator.scope(), enumerator.name(), enumValuesText(enumerator).toLatin1().data());
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
                    retval += ito::RetVal::format(ito::retError, 0, QObject::tr("The key '%s' does not exist in the enumeration %s::%s. " \
                        "Possible keys or values are: %s.").toLatin1().data(),
                        str.toLatin1().data(), enumerator.scope(), enumerator.name(), enumValuesText(enumerator).toLatin1().data());
                    return result;
                }
            }
        }
        else
        {
            retval += ito::RetVal::format(ito::retError, 0, QObject::tr("Use an integer or a string for a value of the enumeration %s::%s").toLatin1().data(),
                enumerator.scope(), enumerator.name());
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
        if (!value.isValid())
        {
            retValue += RetVal::format(retError, 0, QObject::tr("Property '%s' could not be read").toLatin1().data(), propName);
        }
    }
    else
    {
        retValue += RetVal(retError, 0, QObject::tr("Invalid object").toLatin1().data());
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
                retValue += ito::RetVal::format(ito::retError, 0, QObject::tr("Property '%s' is not writeable.").toLatin1().data(), propName);
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
                retValue += ito::RetVal::format(ito::retError, 0, QObject::tr("Property '%s' could not be set. Maybe wrong input type.").toLatin1().data(), propName);
            }

        }
        else
        {
            retValue += ito::RetVal::format(ito::retError, 0, QObject::tr("Property '%s' does not exist.").toLatin1().data(), propName);
        }
    }
    else
    {
        retValue += RetVal(retError, 0, QObject::tr("Invalid object").toLatin1().data());
    }

    return retValue;
}

} //end namespace ito
