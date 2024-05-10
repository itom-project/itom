/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2023, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
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

#include "itomParamManager.h"

#include <qstring.h>
#include <QtWidgets/QStyleOption>
#include <QtWidgets/QStyle>
#include <QtWidgets/QApplication>
#include <qpainter.h>
#include <qcheckbox.h>
#include <qlocale.h>

#include "common/addInInterface.h"
#include "DataObject/dataobj.h"

namespace ito
{

//------------------------------------------------------------------------------
// AbstractParamPropertyManager
class AbstractParamPropertyManagerPrivate
{
    AbstractParamPropertyManager *q_ptr;
    Q_DECLARE_PUBLIC(AbstractParamPropertyManager)
public:

    struct Data
    {
        Data() {}
        Data(const ito::Param &p) : param(p) {}
        ito::Param param;
    };

    typedef QMap<const QtProperty *, Data> PropertyValueMap;
    PropertyValueMap m_values;
    ito::Param m_empty; //this param is returned if no property could not be found in map
};


//------------------------------------------------------------------------------
/*!
    Creates a manager with the given \a parent.
*/
AbstractParamPropertyManager::AbstractParamPropertyManager(QObject *parent)
    : QtAbstractPropertyManager(parent)
{
    d_ptr = new AbstractParamPropertyManagerPrivate;
    d_ptr->q_ptr = this;
}

//------------------------------------------------------------------------------
/*!
    Destroys this manager, and all the properties it has created.
*/
AbstractParamPropertyManager::~AbstractParamPropertyManager()
{
    clear();
    delete d_ptr;
}

//------------------------------------------------------------------------------
const ito::ParamBase &AbstractParamPropertyManager::paramBase(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = d_ptr->m_values.constFind(property);
    if (it == d_ptr->m_values.constEnd())
        return d_ptr->m_empty;
    return it.value().param;
}

//------------------------------------------------------------------------------
const ito::Param &AbstractParamPropertyManager::param(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = d_ptr->m_values.constFind(property);
    if (it == d_ptr->m_values.constEnd())
        return d_ptr->m_empty;
    return it.value().param;
}

//------------------------------------------------------------------------------
/*!
    Returns an icon representing the current state of the given \a
    property.

    The default implementation of this function returns an invalid
    icon.

    \sa QtProperty::valueIcon()
*/
QIcon AbstractParamPropertyManager::valueIcon(const QtProperty *property) const
{
    Q_UNUSED(property)
    return QIcon();
}

//------------------------------------------------------------------------------
/*!
    Returns a string representing the current state of the given \a
    property.

    The default implementation of this function returns an empty
    string.

    \sa QtProperty::valueText()
*/
QString AbstractParamPropertyManager::valueText(const QtProperty *property) const
{
    Q_UNUSED(property)
    return QString();
}

//------------------------------------------------------------------------------
/*!
    \reimp
*/
void AbstractParamPropertyManager::uninitializeProperty(QtProperty *property)
{
    d_ptr->m_values.remove(property);
}


//------------------------------------------------------------------------------
ParamIntPropertyManager::ParamIntPropertyManager(QObject *parent /*= 0*/) :
    AbstractParamPropertyManager(parent)
{
}

//------------------------------------------------------------------------------
ParamIntPropertyManager::~ParamIntPropertyManager()
{
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
void ParamIntPropertyManager::initializeProperty(QtProperty *property)
{
    d_ptr->m_values[property] = AbstractParamPropertyManagerPrivate::Data(ito::Param("", ito::ParamBase::Int, 0, ""));
}

//------------------------------------------------------------------------------
/*!
    \fn void AbstractParamPropertyManager::setValue(QtProperty *property, int value)

    Sets the value of the given \a property to \a value.

    If the specified \a value is not valid according to the given \a
    property's range, the \a value is adjusted to the nearest valid
    value within the range.

    \sa value(), setRange(), valueChanged()
*/
void ParamIntPropertyManager::setParam(QtProperty *property, const ito::Param &param)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = d_ptr->m_values.find(property);
    if (it == d_ptr->m_values.end())
        return;

    Q_ASSERT(param.getType() == ito::ParamBase::Int);

    property->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));

    PrivateData &data = it.value();
    ito::IntMeta *meta = data.param.getMetaT<ito::IntMeta>();
    const ito::IntMeta *metaNew = param.getMetaT<const ito::IntMeta>();

    if ((meta && metaNew && (*meta != *metaNew)) || \
        (meta && !metaNew) || \
        (!meta && metaNew))
    {
        data.param = param;
        emit metaChanged(property, *param.getMetaT<ito::IntMeta>());
        emit valueChanged(property, data.param.getVal<int>());
        emit propertyChanged(property);
    }
    else if (data.param != param)
    {
        data.param.copyValueFrom(&param);
        emit valueChanged(property, data.param.getVal<int>());
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
void ParamIntPropertyManager::setValue(QtProperty *property, int value)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = d_ptr->m_values.find(property);
    if (it == d_ptr->m_values.end())
        return;

    PrivateData &data = it.value();
    if (data.param.getVal<int>() != value)
    {
        data.param.setVal<int>(value);
        emit valueChanged(property, value);
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
/*!
    \reimp
*/
QString ParamIntPropertyManager::valueText(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = d_ptr->m_values.constFind(property);
    if (it == d_ptr->m_values.constEnd())
        return QString();

    const ito::Param &param = it.value().param;
    const ito::IntMeta *meta = param.getMetaT<const ito::IntMeta>();
    if (meta)
    {
        if (meta->getRepresentation() == ito::ParamMeta::Boolean || \
            (meta->getMin() == 0 && meta->getMax() == 1 && meta->getStepSize() == 1))
        {
            static const QString trueText = tr("True");
            static const QString falseText = tr("False");
            return param.getVal<int>() > 0 ? trueText : falseText;
        }
		else if (meta->getRepresentation() == ito::ParamMeta::IPV4Address)
		{
			int val = param.getVal<int>();
			return QString::number(val);
		}
		else if (meta->getRepresentation() == ito::ParamMeta::MACAddress)
		{
			int val = param.getVal<int>();
			return QString::number(val);
		}
		else if (meta->getRepresentation() == ito::ParamMeta::HexNumber)
		{
			int val = param.getVal<int>();
			return QString("0x%1").arg(QString::number(val, 16));
		}
        else if (meta->getUnit().empty() == false)
        {
            return QString("%1 %2").arg(param.getVal<int>()).arg(meta->getUnit().data());
        }
    }

    return QString::number(param.getVal<int>());
}

//------------------------------------------------------------------------------
// Return an icon containing a check box indicator
static QIcon drawCheckBox(bool value)
{
    QStyleOptionButton opt;
    opt.state |= value ? QStyle::State_On : QStyle::State_Off;
    opt.state |= QStyle::State_Enabled;
    const QStyle *style = QApplication::style();
    // Figure out size of an indicator and make sure it is not scaled down in a list view item
    // by making the pixmap as big as a list view icon and centering the indicator in it.
    // (if it is smaller, it can't be helped)
    const int indicatorWidth = style->pixelMetric(QStyle::PM_IndicatorWidth, &opt);
    const int indicatorHeight = style->pixelMetric(QStyle::PM_IndicatorHeight, &opt);
    const int listViewIconSize = indicatorWidth;
    const int pixmapWidth = indicatorWidth;
    const int pixmapHeight = qMax(indicatorHeight, listViewIconSize);

    opt.rect = QRect(0, 0, indicatorWidth, indicatorHeight);
    QPixmap pixmap = QPixmap(pixmapWidth, pixmapHeight);
    pixmap.fill(Qt::transparent);
    {
        // Center?
        const int xoff = (pixmapWidth  > indicatorWidth)  ? (pixmapWidth  - indicatorWidth)  / 2 : 0;
        const int yoff = (pixmapHeight > indicatorHeight) ? (pixmapHeight - indicatorHeight) / 2 : 0;
        QPainter painter(&pixmap);
        painter.translate(xoff, yoff);
        QCheckBox cb;
        style->drawPrimitive(QStyle::PE_IndicatorCheckBox, &opt, &painter, &cb);
    }
    return QIcon(pixmap);
}

//------------------------------------------------------------------------------
/*!
    \reimp
*/
QIcon ParamIntPropertyManager::valueIcon(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = d_ptr->m_values.constFind(property);
    if (it == d_ptr->m_values.constEnd())
        return QIcon();

    const ito::Param &param = it.value().param;
    const ito::IntMeta *meta = param.getMetaT<const ito::IntMeta>();
    if (meta && (meta->getRepresentation() == ito::ParamMeta::Boolean || \
        (meta->getMin() == 0 && meta->getMax() == 1 && meta->getStepSize() == 1)))
    {
        static const QIcon checkedIcon = drawCheckBox(true);
        static const QIcon uncheckedIcon = drawCheckBox(false);
        return param.getVal<int>() > 0 ? checkedIcon : uncheckedIcon;
    }

    return QIcon();
}







//------------------------------------------------------------------------------
ParamCharPropertyManager::ParamCharPropertyManager(QObject *parent /*= 0*/) :
    AbstractParamPropertyManager(parent)
{
}

//------------------------------------------------------------------------------
ParamCharPropertyManager::~ParamCharPropertyManager()
{
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
void ParamCharPropertyManager::initializeProperty(QtProperty *property)
{
    d_ptr->m_values[property] = AbstractParamPropertyManagerPrivate::Data(ito::Param("", ito::ParamBase::Char, 0, ""));
}

//------------------------------------------------------------------------------
/*!
    \fn void AbstractParamPropertyManager::setValue(QtProperty *property, int value)

    Sets the value of the given \a property to \a value.

    If the specified \a value is not valid according to the given \a
    property's range, the \a value is adjusted to the nearest valid
    value within the range.

    \sa value(), setRange(), valueChanged()
*/
void ParamCharPropertyManager::setParam(QtProperty *property, const ito::Param &param)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = d_ptr->m_values.find(property);
    if (it == d_ptr->m_values.end())
        return;

    Q_ASSERT(param.getType() == ito::ParamBase::Char);

    property->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));

    PrivateData &data = it.value();
    ito::CharMeta *meta = data.param.getMetaT<ito::CharMeta>();
    const ito::CharMeta *metaNew = param.getMetaT<const ito::CharMeta>();

    if ((meta && metaNew && (*meta != *metaNew)) || \
        (meta && !metaNew) || \
        (!meta && metaNew))
    {
        data.param = param;
        emit metaChanged(property, *param.getMetaT<ito::CharMeta>());
        emit valueChanged(property, data.param.getVal<char>());
        emit propertyChanged(property);
    }
    else if (data.param != param)
    {
        data.param.copyValueFrom(&param);
        emit valueChanged(property, data.param.getVal<char>());
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
void ParamCharPropertyManager::setValue(QtProperty *property, char value)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = d_ptr->m_values.find(property);
    if (it == d_ptr->m_values.end())
        return;

    PrivateData &data = it.value();
    if (data.param.getVal<char>() != value)
    {
        data.param.setVal<char>(value);
        emit valueChanged(property, value);
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
/*!
    \reimp
*/
QString ParamCharPropertyManager::valueText(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = d_ptr->m_values.constFind(property);
    if (it == d_ptr->m_values.constEnd())
        return QString();

    const ito::Param &param = it.value().param;
    const ito::CharMeta *meta = param.getMetaT<const ito::CharMeta>();
    if (meta)
    {
        if (meta->getRepresentation() == ito::ParamMeta::Boolean || \
            (meta->getMin() == 0 && meta->getMax() == 1 && meta->getStepSize() == 1))
        {
            static const QString trueText = tr("True");
            static const QString falseText = tr("False");
            return param.getVal<int>() > 0 ? trueText : falseText;
        }
        else if (meta->getUnit().empty() == false)
        {
            return QString("%1 %2").arg(param.getVal<char>()).arg(meta->getUnit().data());
        }
    }

    return QString::number(param.getVal<char>());
}

//------------------------------------------------------------------------------
/*!
    \reimp
*/
QIcon ParamCharPropertyManager::valueIcon(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = d_ptr->m_values.constFind(property);
    if (it == d_ptr->m_values.constEnd())
        return QIcon();

    const ito::Param &param = it.value().param;
    const ito::IntMeta *meta = param.getMetaT<const ito::IntMeta>();
    if (meta && (meta->getRepresentation() == ito::ParamMeta::Boolean || \
        (meta->getMin() == 0 && meta->getMax() == 1 && meta->getStepSize() == 1)))
    {
        static const QIcon checkedIcon = drawCheckBox(true);
        static const QIcon uncheckedIcon = drawCheckBox(false);
        return param.getVal<char>() > 0 ? checkedIcon : uncheckedIcon;
    }

    return QIcon();
}



//------------------------------------------------------------------------------
ParamDoublePropertyManager::ParamDoublePropertyManager(QObject *parent /*= 0*/) :
    AbstractParamPropertyManager(parent),
    m_popupSlider(false)
{
}

//------------------------------------------------------------------------------
ParamDoublePropertyManager::~ParamDoublePropertyManager()
{
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
void ParamDoublePropertyManager::initializeProperty(QtProperty *property)
{
    d_ptr->m_values[property] = AbstractParamPropertyManagerPrivate::Data(ito::Param("", ito::ParamBase::Double, 0.0, NULL, NULL));
}

//------------------------------------------------------------------------------
/*!
    \fn void AbstractParamPropertyManager::setValue(QtProperty *property, int value)

    Sets the value of the given \a property to \a value.

    If the specified \a value is not valid according to the given \a
    property's range, the \a value is adjusted to the nearest valid
    value within the range.

    \sa value(), setRange(), valueChanged()
*/
void ParamDoublePropertyManager::setParam(QtProperty *property, const ito::Param &param)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = d_ptr->m_values.find(property);
    if (it == d_ptr->m_values.end())
        return;

    Q_ASSERT(param.getType() == ito::ParamBase::Double);

    property->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));

    PrivateData &data = it.value();
    ito::DoubleMeta *meta = data.param.getMetaT<ito::DoubleMeta>();
    const ito::DoubleMeta *metaNew = param.getMetaT<const ito::DoubleMeta>();

    if ((meta && metaNew && (*meta != *metaNew)) || \
        (meta && !metaNew) || \
        (!meta && metaNew))
    {
        data.param = param;
        emit metaChanged(property, *param.getMetaT<ito::DoubleMeta>());
        emit valueChanged(property, data.param.getVal<ito::float64>());
        emit propertyChanged(property);
    }
    else if (data.param != param)
    {
        data.param.copyValueFrom(&param);
        emit valueChanged(property, data.param.getVal<ito::float64>());
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
void ParamDoublePropertyManager::setValue(QtProperty *property, double value)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = d_ptr->m_values.find(property);
    if (it == d_ptr->m_values.end())
        return;

    PrivateData &data = it.value();
    if (data.param.getVal<ito::float64>() != value)
    {
        data.param.setVal<ito::float64>(value);
        emit valueChanged(property, value);
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
/*!
    \reimp
*/
QString ParamDoublePropertyManager::valueText(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = d_ptr->m_values.constFind(property);
    if (it == d_ptr->m_values.constEnd())
        return QString();

    const ito::Param &param = it.value().param;
    const ito::DoubleMeta *meta = param.getMetaT<const ito::DoubleMeta>();
    QString number;
    QLocale locale;

    if (meta)
    {
        double val = param.getVal<ito::float64>();

        switch (meta->getDisplayNotation())
        {
        case ito::DoubleMeta::Automatic:
            if (std::abs(val) > 100000)
            {
                number = locale.toString(param.getVal<ito::float64>(), 'e', meta->getDisplayPrecision());
            }
            else
            {
                number = locale.toString(param.getVal<ito::float64>(), 'f', meta->getDisplayPrecision());
            }
            break;
        case ito::DoubleMeta::Fixed:
            number = locale.toString(param.getVal<ito::float64>(), 'f', meta->getDisplayPrecision());
            break;
        case ito::DoubleMeta::Scientific:
            number = locale.toString(param.getVal<ito::float64>(), 'e', meta->getDisplayPrecision());
            break;
        }
    }
    else
    {
        number= locale.toString(param.getVal<ito::float64>());
    }

    if (meta)
    {
        if (meta->getRepresentation() == ito::ParamMeta::Boolean || \
            (meta->getMin() == 0 && meta->getMax() == 1 && meta->getStepSize() == 1))
        {
            static const QString trueText = tr("True");
            static const QString falseText = tr("False");
            return param.getVal<int>() > 0 ? trueText : falseText;
        }
        else if (meta->getUnit().empty() == false)
        {
            return QString("%1 %2").arg(number).arg(meta->getUnit().data());
        }
    }

    return number;
}

//------------------------------------------------------------------------------
/*!
    \reimp
*/
QIcon ParamDoublePropertyManager::valueIcon(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = d_ptr->m_values.constFind(property);
    if (it == d_ptr->m_values.constEnd())
        return QIcon();

    const ito::Param &param = it.value().param;
    const ito::DoubleMeta *meta = param.getMetaT<const ito::DoubleMeta>();
    if (meta && (meta->getRepresentation() == ito::ParamMeta::Boolean || \
        (qFuzzyCompare(meta->getMin(), 0.0) && qFuzzyCompare(meta->getMax(), 1.0) && qFuzzyCompare(meta->getStepSize(), 1.0))))
    {
        static const QIcon checkedIcon = drawCheckBox(true);
        static const QIcon uncheckedIcon = drawCheckBox(false);
        return param.getVal<char>() > 0 ? checkedIcon : uncheckedIcon;
    }

    return QIcon();
}

//------------------------------------------------------------------------------
bool ParamDoublePropertyManager::hasPopupSlider() const
{
    return m_popupSlider;
}

//------------------------------------------------------------------------------
void ParamDoublePropertyManager::setPopupSlider(bool popup)
{
    m_popupSlider = popup;
}

//------------------------------------------------------------------------------
ParamStringPropertyManager::ParamStringPropertyManager(QObject *parent /*= 0*/) :
AbstractParamPropertyManager(parent)
{
}

//------------------------------------------------------------------------------
ParamStringPropertyManager::~ParamStringPropertyManager()
{
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
void ParamStringPropertyManager::initializeProperty(QtProperty *property)
{
    d_ptr->m_values[property] = AbstractParamPropertyManagerPrivate::Data(ito::Param("", ito::ParamBase::String, "", ""));
}

//------------------------------------------------------------------------------
/*!
\fn void AbstractParamPropertyManager::setValue(QtProperty *property, int value)

Sets the value of the given \a property to \a value.

If the specified \a value is not valid according to the given \a
property's range, the \a value is adjusted to the nearest valid
value within the range.

\sa value(), setRange(), valueChanged()
*/
void ParamStringPropertyManager::setParam(QtProperty *property, const ito::Param &param)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = d_ptr->m_values.find(property);
    if (it == d_ptr->m_values.end())
        return;

    Q_ASSERT(param.getType() == ito::ParamBase::String);

    property->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));

    PrivateData &data = it.value();
    ito::StringMeta *meta = data.param.getMetaT<ito::StringMeta>();
    const ito::StringMeta *metaNew = param.getMetaT<const ito::StringMeta>();

    if ((meta && metaNew && (*meta != *metaNew)) || \
        (meta && !metaNew) || \
        (!meta && metaNew))
    {
        data.param = param;
        emit metaChanged(property, *param.getMetaT<ito::StringMeta>());
        emit valueChanged(property, data.param.getVal<const char*>());
        emit propertyChanged(property);
    }
    else if (data.param != param)
    {
        data.param.copyValueFrom(&param);
        emit valueChanged(property, data.param.getVal<const char*>());
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
void ParamStringPropertyManager::setValue(QtProperty *property, const QByteArray &value)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = d_ptr->m_values.find(property);
    if (it == d_ptr->m_values.end())
        return;

    PrivateData &data = it.value();
    if (value != QByteArray(data.param.getVal<const char*>()))
    {
        data.param.setVal<const char*>(value.data());
        emit valueChanged(property, value);
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
QString ParamStringPropertyManager::valueText(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = d_ptr->m_values.constFind(property);
    if (it == d_ptr->m_values.constEnd())
        return QString();

    const ito::Param &param = it.value().param;
    return QLatin1String(param.getVal<const char*>());
}



//------------------------------------------------------------------------------
ParamIntervalPropertyManager::ParamIntervalPropertyManager(QObject *parent /*= 0*/) :
AbstractParamPropertyManager(parent)
{
}

//------------------------------------------------------------------------------
ParamIntervalPropertyManager::~ParamIntervalPropertyManager()
{
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
void ParamIntervalPropertyManager::initializeProperty(QtProperty *property)
{
    int vals[] = { 0, 0 };
    d_ptr->m_values[property] = AbstractParamPropertyManagerPrivate::Data(ito::Param("", ito::ParamBase::IntArray, 2, vals, ""));
}

//------------------------------------------------------------------------------
/*!
\fn void AbstractParamPropertyManager::setValue(QtProperty *property, int value)

Sets the value of the given \a property to \a value.

If the specified \a value is not valid according to the given \a
property's range, the \a value is adjusted to the nearest valid
value within the range.

\sa value(), setRange(), valueChanged()
*/
void ParamIntervalPropertyManager::setParam(QtProperty *property, const ito::Param &param)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = d_ptr->m_values.find(property);
    if (it == d_ptr->m_values.end())
        return;

    Q_ASSERT(param.getType() == (ito::ParamBase::IntArray));

    property->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));

    PrivateData &data = it.value();
    ito::IntervalMeta *meta = data.param.getMetaT<ito::IntervalMeta>();
    const ito::IntervalMeta *metaNew = param.getMetaT<const ito::IntervalMeta>();

    if ((meta && metaNew && (*meta != *metaNew)) || \
        (meta && !metaNew) || \
        (!meta && metaNew))
    {
        data.param = param;
        emit metaChanged(property, *param.getMetaT<ito::IntervalMeta>());
        const int* vals = data.param.getVal<const int*>();
        emit valueChanged(property, vals[0], vals[1]);
        emit propertyChanged(property);
    }
    else if (data.param != param)
    {
        if (data.param.getType() != param.getType())
        {
            data.param = param;
        }
        else
        {
            data.param.copyValueFrom(&param);
        }
        const int* vals = data.param.getVal<const int*>();
        emit valueChanged(property, vals[0], vals[1]);
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
void ParamIntervalPropertyManager::setValue(QtProperty *property, int min, int max)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = d_ptr->m_values.find(property);
    if (it == d_ptr->m_values.end())
        return;

    PrivateData &data = it.value();
    int* vals = data.param.getVal<int*>();
    int len = data.param.getLen();

    if (len != 2 || vals[0] != min || vals[1] != max)
    {
        if (len == 2)
        {
            vals[0] = min; vals[1] = max;
        }
        else
        {
            int v[] = {min, max};
            data.param.setVal<int*>(v, 2);
        }
        emit valueChanged(property, min, max);
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
QString ParamIntervalPropertyManager::valueText(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = d_ptr->m_values.constFind(property);
    if (it == d_ptr->m_values.constEnd())
        return QString();

    const ito::Param &param = it.value().param;
    const ito::IntervalMeta *meta = param.getMetaT<const ito::IntervalMeta>();

    if (meta && meta->isIntervalNotRange())
    {
        return QString("[%1,%2)").arg(param.getVal<int*>()[0]).arg(param.getVal<int*>()[1]);
    }
    else
    {
        const int* vals = param.getVal<const int*>();
        return QString("[%1,%2], size: %3").arg(vals[0]).arg(vals[1]).arg(vals[1] - vals[0] + 1);
    }
}

//------------------------------------------------------------------------------
// AbstractParamPropertyManager
class ParamRectPropertyManagerPrivate
{
    ParamRectPropertyManager *q_ptr;
    Q_DECLARE_PUBLIC(ParamRectPropertyManager)
public:

    void slotIntervalChanged(QtProperty *property, int min, int max);
    void slotPropertyDestroyed(QtProperty *property);
    void setMeta(QtProperty *property, const ito::RectMeta &meta);

    ito::ParamIntervalPropertyManager *m_intervalPropertyManager;
    ito::AbstractParamPropertyManagerPrivate *m_d_ptr;

    QMap<const QtProperty*, QtProperty*> m_propertyToWidth;
    QMap<const QtProperty*, QtProperty*> m_propertyToHeight;

    QMap<const QtProperty*, QtProperty*> m_widthToProperty;
    QMap<const QtProperty*, QtProperty*> m_heightToProperty;
};

void ParamRectPropertyManagerPrivate::slotIntervalChanged(QtProperty *property, int min, int max)
{
    if (QtProperty *prop = m_widthToProperty.value(property, 0))
    {
        const int* vals = m_d_ptr->m_values[prop].param.getVal<const int*>();
        q_ptr->setValue(prop, min, vals[1], 1 + max - min, vals[3]);
    }
    else if (QtProperty *prop = m_heightToProperty.value(property))
    {
        const int* vals = m_d_ptr->m_values[prop].param.getVal<const int*>();
        q_ptr->setValue(prop, vals[0], min, vals[2], 1 + max - min);
    }
}

void ParamRectPropertyManagerPrivate::slotPropertyDestroyed(QtProperty *property)
{
    if (QtProperty *pointProp = m_widthToProperty.value(property, 0)) {
        m_propertyToWidth[pointProp] = 0;
        m_widthToProperty.remove(property);
    } else if (QtProperty *pointProp = m_heightToProperty.value(property, 0)) {
        m_propertyToHeight[pointProp] = 0;
        m_heightToProperty.remove(property);
    }
}

//------------------------------------------------------------------------------
ParamRectPropertyManager::ParamRectPropertyManager(QObject *parent /*= 0*/) :
AbstractParamPropertyManager(parent)
{
    d_ptr = new ParamRectPropertyManagerPrivate;
    d_ptr->q_ptr = this;
    d_ptr->m_d_ptr = AbstractParamPropertyManager::d_ptr;

    d_ptr->m_intervalPropertyManager = new ito::ParamIntervalPropertyManager(this);
    connect(d_ptr->m_intervalPropertyManager, SIGNAL(valueChanged(QtProperty *, int, int)),
                this, SLOT(slotIntervalChanged(QtProperty *, int, int)));
    connect(d_ptr->m_intervalPropertyManager, SIGNAL(propertyDestroyed(QtProperty *)),
                this, SLOT(slotPropertyDestroyed(QtProperty *)));
}

//------------------------------------------------------------------------------
ParamRectPropertyManager::~ParamRectPropertyManager()
{
    delete d_ptr;
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
void ParamRectPropertyManager::initializeProperty(QtProperty *property)
{
    int vals[] = {0, 0, 0, 0};
    AbstractParamPropertyManager::d_ptr->m_values[property] = AbstractParamPropertyManagerPrivate::Data(ito::Param("", ito::ParamBase::IntArray, 4, vals, ""));
    QtProperty *widthProp = d_ptr->m_intervalPropertyManager->addProperty();
    widthProp->setPropertyName(tr("width"));
    d_ptr->m_intervalPropertyManager->setValue(widthProp, 0, 0);
    d_ptr->m_propertyToWidth[property] = widthProp;
    d_ptr->m_widthToProperty[widthProp] = property;
    property->addSubProperty(widthProp);

    QtProperty *heightProp = d_ptr->m_intervalPropertyManager->addProperty();
    heightProp->setPropertyName(tr("height"));
    d_ptr->m_intervalPropertyManager->setValue(heightProp, 0, 0);
    d_ptr->m_propertyToHeight[property] = heightProp;
    d_ptr->m_heightToProperty[heightProp] = property;
    property->addSubProperty(heightProp);
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
void ParamRectPropertyManager::uninitializeProperty(QtProperty *property)
{
    QtProperty *widthProp = d_ptr->m_propertyToWidth[property];
    if (widthProp) {
        d_ptr->m_widthToProperty.remove(widthProp);
        delete widthProp;
    }
    d_ptr->m_propertyToWidth.remove(property);

    QtProperty *yProp = d_ptr->m_propertyToHeight[property];
    if (yProp) {
        d_ptr->m_heightToProperty.remove(yProp);
        delete yProp;
    }
    d_ptr->m_propertyToHeight.remove(property);
}

//------------------------------------------------------------------------------
/*!
\fn void AbstractParamPropertyManager::setValue(QtProperty *property, int value)

Sets the value of the given \a property to \a value.

If the specified \a value is not valid according to the given \a
property's range, the \a value is adjusted to the nearest valid
value within the range.

\sa value(), setRange(), valueChanged()
*/
void ParamRectPropertyManager::setParam(QtProperty *property, const ito::Param &param)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = AbstractParamPropertyManager::d_ptr->m_values.find(property);
    if (it == AbstractParamPropertyManager::d_ptr->m_values.end())
        return;

    Q_ASSERT(param.getType() == (ito::ParamBase::IntArray));

    property->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));

    PrivateData &data = it.value();
    ito::RectMeta *meta = data.param.getMetaT<ito::RectMeta>();
    const ito::RectMeta *metaNew = param.getMetaT<const ito::RectMeta>();

    if ((meta && metaNew && (*meta != *metaNew)) || \
        (meta && !metaNew) || \
        (!meta && metaNew))
    {
        data.param = param;
        const int* vals = data.param.getVal<const int*>();
        int vw[] = {vals[0], vals[0] + vals[2] - 1};
        ito::Param width("width", ito::ParamBase::IntArray, 2, vw, new ito::RangeMeta(metaNew->getWidthRangeMeta()), "");
        d_ptr->m_intervalPropertyManager->setParam(d_ptr->m_propertyToWidth[property], width);
        int vh[] = {vals[1], vals[1] + vals[3] - 1};
        ito::Param height("height", ito::ParamBase::IntArray, 2, vh, new ito::RangeMeta(metaNew->getHeightRangeMeta()), "");
        d_ptr->m_intervalPropertyManager->setParam(d_ptr->m_propertyToHeight[property], height);
        emit metaChanged(property, *metaNew);
        emit valueChanged(property, vals[0], vals[1], vals[2], vals[3]);
        emit propertyChanged(property);
    }
    else if (data.param != param)
    {
        data.param.copyValueFrom(&param);
        const int* vals = data.param.getVal<const int*>();

        d_ptr->m_intervalPropertyManager->setValue(d_ptr->m_propertyToWidth[property], vals[0], vals[0] + vals[2] - 1);
        d_ptr->m_intervalPropertyManager->setValue(d_ptr->m_propertyToHeight[property], vals[1], vals[1] + vals[3] - 1);

        emit valueChanged(property, vals[0], vals[1], vals[2], vals[3]);
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
void ParamRectPropertyManager::setValue(QtProperty *property, int left, int top, int width, int height)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = AbstractParamPropertyManager::d_ptr->m_values.find(property);
    if (it == AbstractParamPropertyManager::d_ptr->m_values.end())
        return;

    PrivateData &data = it.value();
    int* vals = data.param.getVal<int*>();
    int len = data.param.getLen();

    if (len != 4 || vals[0] != left || vals[1] != top ||  \
        vals[2] != width || vals[3] != height)
    {
        if (len == 4)
        {
            vals[0] = left; vals[1] = top;
            vals[2] = width; vals[3] = height;
        }
        else
        {
            int v[] = {left, top, width, height};
            data.param.setVal<int*>(v, 4);
        }
        emit valueChanged(property, left, top, width, height);
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
QString ParamRectPropertyManager::valueText(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = AbstractParamPropertyManager::d_ptr->m_values.constFind(property);
    if (it == AbstractParamPropertyManager::d_ptr->m_values.constEnd())
        return QString();

    const ito::Param &param = it.value().param;
    const int* vals = param.getVal<const int*>();
    return QString("x0:%1 y0:%2 w:%3 h:%4").arg(vals[0]).arg(vals[1]).arg(vals[2]).arg(vals[3]);
}

/*!
Returns the manager that creates the nested \e width and \e height subproperties.

In order to provide editing widgets for the mentioned
subproperties in a property browser widget, this manager must be
associated with an editor factory.
*/
ParamIntervalPropertyManager *ParamRectPropertyManager::subIntervalPropertyManager() const
{
    return d_ptr->m_intervalPropertyManager;
}





//------------------------------------------------------------------------------
// ParamCharArrayPropertyManagerPrivate
class ParamCharArrayPropertyManagerPrivate
{
    ParamCharArrayPropertyManager *q_ptr;
    Q_DECLARE_PUBLIC(ParamCharArrayPropertyManager)
public:

    void slotValueChanged(QtProperty *property, char value);
    void slotPropertyDestroyed(QtProperty *property);
    void setMeta(QtProperty *property, const ito::CharArrayMeta &meta);

    ito::ParamCharPropertyManager *m_numberPropertyManager;
    ito::AbstractParamPropertyManagerPrivate *m_d_ptr;

    QMap<const QtProperty*, QList<QtProperty*> > m_propertyToValues;
    QMap<const QtProperty*, QtProperty*> m_valuesToProperty;
};

void ParamCharArrayPropertyManagerPrivate::slotValueChanged(QtProperty *property, char value)
{
    if (QtProperty *prop = m_valuesToProperty.value(property, 0))
    {
        if (m_propertyToValues.contains(prop))
        {
            int index = m_propertyToValues[prop].indexOf(property);
            int num = m_d_ptr->m_values[prop].param.getLen();
            if (index >= 0 && index < num)
            {
                const ParamCharArrayPropertyManager::DataType* vals = m_d_ptr->m_values[prop].param.getVal<const ParamCharArrayPropertyManager::DataType*>();
                ParamCharArrayPropertyManager::DataType* newvals = (ParamCharArrayPropertyManager::DataType*)malloc(num * sizeof(ParamCharArrayPropertyManager::DataType));
                memcpy(newvals, vals, sizeof(ParamCharArrayPropertyManager::DataType) * num);
                newvals[index] = value;
                q_ptr->setValue(prop, num, newvals);
                free(newvals);
            }
        }
    }
}

void ParamCharArrayPropertyManagerPrivate::slotPropertyDestroyed(QtProperty *property)
{
    if (QtProperty *pointProp = m_valuesToProperty.value(property, 0))
    {
        m_propertyToValues[pointProp].removeOne(property);
        if (m_propertyToValues[pointProp].size() == 0)
        {
            m_propertyToValues.remove(pointProp);
            m_valuesToProperty.remove(property);
        }
    }
}

//------------------------------------------------------------------------------
ParamCharArrayPropertyManager::ParamCharArrayPropertyManager(QObject *parent /*= 0*/) :
AbstractParamPropertyManager(parent)
{
    d_ptr = new ParamCharArrayPropertyManagerPrivate;
    d_ptr->q_ptr = this;
    d_ptr->m_d_ptr = AbstractParamPropertyManager::d_ptr;

    d_ptr->m_numberPropertyManager = new ito::ParamCharPropertyManager(this);
    connect(d_ptr->m_numberPropertyManager, SIGNAL(valueChanged(QtProperty *, char)),
        this, SLOT(slotValueChanged(QtProperty *, char)));
    connect(d_ptr->m_numberPropertyManager, SIGNAL(propertyDestroyed(QtProperty *)),
        this, SLOT(slotPropertyDestroyed(QtProperty *)));
}

//------------------------------------------------------------------------------
ParamCharArrayPropertyManager::~ParamCharArrayPropertyManager()
{
    delete d_ptr;
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
void ParamCharArrayPropertyManager::initializeProperty(QtProperty *property)
{
    char *ptr = NULL;
    AbstractParamPropertyManager::d_ptr->m_values[property] = AbstractParamPropertyManagerPrivate::Data(ito::Param("", ito::ParamBase::CharArray, 0, ptr, NULL, ""));
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
void ParamCharArrayPropertyManager::uninitializeProperty(QtProperty *property)
{
    foreach(QtProperty* prop, d_ptr->m_propertyToValues[property])
    {
        d_ptr->m_valuesToProperty.remove(prop);
        delete prop;
    }
    d_ptr->m_propertyToValues.remove(property);
}

//------------------------------------------------------------------------------
/*!
Sets the value of the given \a property to \a value.

If the specified \a value is not valid according to the given \a
property's range, the \a value is adjusted to the nearest valid
value within the range.

\sa value(), setRange(), valueChanged()
*/
void ParamCharArrayPropertyManager::setParam(QtProperty *property, const ito::Param &param)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = AbstractParamPropertyManager::d_ptr->m_values.find(property);
    if (it == AbstractParamPropertyManager::d_ptr->m_values.end())
        return;

    Q_ASSERT(param.getType() == (ito::ParamBase::CharArray));

    property->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));

    PrivateData &data = it.value();
    ito::CharArrayMeta *meta = data.param.getMetaT<ito::CharArrayMeta>();
    const ito::CharArrayMeta *metaNew = param.getMetaT<const ito::CharArrayMeta>();
    int len = std::max(0, param.getLen());
    const DataType* values = param.getVal<const DataType*>();

    int oldSubItems = d_ptr->m_propertyToValues.contains(property) ? d_ptr->m_propertyToValues[property].size() : 0;

    if (oldSubItems > len)
    {
        //remove supernumerous sub-items
        for (int i = oldSubItems; i = len; --i)
        {
            QtProperty *subProp = d_ptr->m_propertyToValues[property][i - 1];
            property->removeSubProperty(subProp);
            d_ptr->m_valuesToProperty.remove(subProp);
            delete subProp;
            d_ptr->m_propertyToValues[property].removeLast();
        }
    }

    if (len > oldSubItems)
    {
        //add new sub-items
        for (int i = oldSubItems; i < len; ++i)
        {
            QtProperty *prop = d_ptr->m_numberPropertyManager->addProperty();
            prop->setPropertyName(tr("%1").arg(i));
            d_ptr->m_numberPropertyManager->setValue(prop, 0);
            d_ptr->m_propertyToValues[property].append(prop);
            d_ptr->m_valuesToProperty[prop] = property;
            property->addSubProperty(prop);
        }
    }

    if ((meta && metaNew && (*meta != *metaNew)) || \
        (meta && !metaNew) || \
        (!meta && metaNew))
    {
        data.param = param;

        const DataType* vals = data.param.getVal<const DataType*>();
        ito::CharMeta *im;

        for (int i = 0; i < len; ++i)
        {
            im = new ito::CharMeta(metaNew->getMin(), metaNew->getMax(), metaNew->getStepSize(), metaNew->getCategory());
            ito::Param val("val", ito::ParamBase::Char, vals[i], im, "");
            d_ptr->m_numberPropertyManager->setParam(d_ptr->m_propertyToValues[property][i], val);
        }

        emit metaChanged(property, *metaNew);
        emit valueChanged(property, len, vals);
        emit propertyChanged(property);
    }
    else if (data.param != param)
    {
        data.param.copyValueFrom(&param);
        const DataType* vals = data.param.getVal<const DataType*>();
        ito::CharMeta *im;

        for (int i = 0; i < len; ++i)
        {
            if (metaNew)
            {
                im = new ito::CharMeta(metaNew->getMin(), metaNew->getMax(), metaNew->getStepSize(), metaNew->getCategory());
            }
            else
            {
                im = NULL;
            }
            ito::Param val("val", ito::ParamBase::Char, vals[i], im, "");
            d_ptr->m_numberPropertyManager->setParam(d_ptr->m_propertyToValues[property][i], val);
        }

        emit valueChanged(property, len, data.param.getVal<const DataType*>());
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
void ParamCharArrayPropertyManager::setValue(QtProperty *property, int num, const char* values)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = AbstractParamPropertyManager::d_ptr->m_values.find(property);
    if (it == AbstractParamPropertyManager::d_ptr->m_values.end())
        return;

    PrivateData &data = it.value();
    DataType* vals = data.param.getVal<DataType*>();
    int len = data.param.getLen();

    if (len != num || memcmp(values, vals, num * sizeof(DataType)) != 0)
    {
        data.param.setVal<const DataType*>(values, num);

        for (int i = 0; i < d_ptr->m_propertyToValues[property].size(); ++i)
        {
            d_ptr->m_numberPropertyManager->setValue(d_ptr->m_propertyToValues[property][i], values[i]);
        }

        emit valueChanged(property, num, values);
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
QString ParamCharArrayPropertyManager::valueText(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = AbstractParamPropertyManager::d_ptr->m_values.constFind(property);
    if (it == AbstractParamPropertyManager::d_ptr->m_values.constEnd())
        return QString();

    const ito::Param &param = it.value().param;
    int len = param.getLen();
    const DataType* vals = param.getVal<const DataType*>();

    switch (len)
    {
    case 0:
        return tr("<empty>");
    case 1:
        return QString("[%1]").arg(vals[0]);
    case 2:
        return QString("[%1,%2]").arg(vals[0]).arg(vals[1]);
    case 3:
        return QString("[%1,%2,%3]").arg(vals[0]).arg(vals[1]).arg(vals[2]);
    default:
        return tr("<%1 values>").arg(len);
    }
}

/*!
Returns the manager that creates the nested subproperties.

In order to provide editing widgets for the mentioned
subproperties in a property browser widget, this manager must be
associated with an editor factory.
*/
ParamCharPropertyManager *ParamCharArrayPropertyManager::subPropertyManager() const
{
    return d_ptr->m_numberPropertyManager;
}









//------------------------------------------------------------------------------
// ParamIntArrayPropertyManagerPrivate
class ParamIntArrayPropertyManagerPrivate
{
    ParamIntArrayPropertyManager *q_ptr;
    Q_DECLARE_PUBLIC(ParamIntArrayPropertyManager)
public:

    void slotValueChanged(QtProperty *property, int value);
    void slotPropertyDestroyed(QtProperty *property);
    void setMeta(QtProperty *property, const ito::IntArrayMeta &meta);

    ito::ParamIntPropertyManager *m_numberPropertyManager;
    ito::AbstractParamPropertyManagerPrivate *m_d_ptr;

    QMap<const QtProperty*, QList<QtProperty*> > m_propertyToValues;
    QMap<const QtProperty*, QtProperty*> m_valuesToProperty;
};

void ParamIntArrayPropertyManagerPrivate::slotValueChanged(QtProperty *property, ito::int32 value)
{
    if (QtProperty *prop = m_valuesToProperty.value(property, 0))
    {
        if (m_propertyToValues.contains(prop))
        {
            int index = m_propertyToValues[prop].indexOf(property);
            int num = m_d_ptr->m_values[prop].param.getLen();
            if (index >= 0 && index < num)
            {
                const ParamIntArrayPropertyManager::DataType* vals = m_d_ptr->m_values[prop].param.getVal<const ParamIntArrayPropertyManager::DataType*>();
                ParamIntArrayPropertyManager::DataType* newvals = (ParamIntArrayPropertyManager::DataType*)malloc(num * sizeof(ParamIntArrayPropertyManager::DataType));
                memcpy(newvals, vals, sizeof(ParamIntArrayPropertyManager::DataType) * num);
                newvals[index] = value;
                q_ptr->setValue(prop, num, newvals);
                free(newvals);
            }
        }
    }
}

void ParamIntArrayPropertyManagerPrivate::slotPropertyDestroyed(QtProperty *property)
{
    if (QtProperty *pointProp = m_valuesToProperty.value(property, 0))
    {
        m_propertyToValues[pointProp].removeOne(property);
        if (m_propertyToValues[pointProp].size() == 0)
        {
            m_propertyToValues.remove(pointProp);
            m_valuesToProperty.remove(property);
        }
    }
}

//------------------------------------------------------------------------------
ParamIntArrayPropertyManager::ParamIntArrayPropertyManager(QObject *parent /*= 0*/) :
AbstractParamPropertyManager(parent)
{
    d_ptr = new ParamIntArrayPropertyManagerPrivate;
    d_ptr->q_ptr = this;
    d_ptr->m_d_ptr = AbstractParamPropertyManager::d_ptr;

    d_ptr->m_numberPropertyManager = new ito::ParamIntPropertyManager(this);
    connect(d_ptr->m_numberPropertyManager, SIGNAL(valueChanged(QtProperty *, int)),
        this, SLOT(slotValueChanged(QtProperty *, int)));
    connect(d_ptr->m_numberPropertyManager, SIGNAL(propertyDestroyed(QtProperty *)),
        this, SLOT(slotPropertyDestroyed(QtProperty *)));
}

//------------------------------------------------------------------------------
ParamIntArrayPropertyManager::~ParamIntArrayPropertyManager()
{
    delete d_ptr;
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
void ParamIntArrayPropertyManager::initializeProperty(QtProperty *property)
{
    DataType *ptr = NULL;
    AbstractParamPropertyManager::d_ptr->m_values[property] = AbstractParamPropertyManagerPrivate::Data(ito::Param("", ito::ParamBase::IntArray, 0, ptr, NULL, ""));
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
void ParamIntArrayPropertyManager::uninitializeProperty(QtProperty *property)
{
    foreach(QtProperty* prop, d_ptr->m_propertyToValues[property])
    {
        d_ptr->m_valuesToProperty.remove(prop);
        delete prop;
    }
    d_ptr->m_propertyToValues.remove(property);
}

//------------------------------------------------------------------------------
/*!
Sets the value of the given \a property to \a value.

If the specified \a value is not valid according to the given \a
property's range, the \a value is adjusted to the nearest valid
value within the range.

\sa value(), setRange(), valueChanged()
*/
void ParamIntArrayPropertyManager::setParam(QtProperty *property, const ito::Param &param)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = AbstractParamPropertyManager::d_ptr->m_values.find(property);
    if (it == AbstractParamPropertyManager::d_ptr->m_values.end())
        return;

    Q_ASSERT(param.getType() == (ito::ParamBase::IntArray));

    property->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));

    PrivateData &data = it.value();
    ito::IntArrayMeta *meta = data.param.getMetaT<ito::IntArrayMeta>();
    const ito::IntArrayMeta *metaNew = param.getMetaT<const ito::IntArrayMeta>();
    int len = std::max(0, param.getLen());
    const DataType* values = param.getVal<const DataType*>();

    int oldSubItems = d_ptr->m_propertyToValues.contains(property) ? d_ptr->m_propertyToValues[property].size() : 0;

    if (oldSubItems > len)
    {
        //remove supernumerous sub-items
        for (int i = oldSubItems; i = len; --i)
        {
            QtProperty *subProp = d_ptr->m_propertyToValues[property][i - 1];
            property->removeSubProperty(subProp);
            d_ptr->m_valuesToProperty.remove(subProp);
            delete subProp;
            d_ptr->m_propertyToValues[property].removeLast();
        }
    }

    if (len > oldSubItems)
    {
        //add new sub-items
        for (int i = oldSubItems; i < len; ++i)
        {
            QtProperty *prop = d_ptr->m_numberPropertyManager->addProperty();
            prop->setPropertyName(tr("%1").arg(i));
            d_ptr->m_numberPropertyManager->setValue(prop, 0);
            d_ptr->m_propertyToValues[property].append(prop);
            d_ptr->m_valuesToProperty[prop] = property;
            property->addSubProperty(prop);
        }
    }

    if ((meta && metaNew && (*meta != *metaNew)) || \
        (meta && !metaNew) || \
        (!meta && metaNew))
    {
        data.param = param;

        const DataType* vals = data.param.getVal<const DataType*>();
        ito::IntMeta *im;

        for (int i = 0; i < len; ++i)
        {
            im = new ito::IntMeta(metaNew->getMin(), metaNew->getMax(), metaNew->getStepSize(), metaNew->getCategory());
            ito::Param val("val", ito::ParamBase::Int, vals[i], im, "");
            d_ptr->m_numberPropertyManager->setParam(d_ptr->m_propertyToValues[property][i], val);
        }

        emit metaChanged(property, *metaNew);
        emit valueChanged(property, len, vals);
        emit propertyChanged(property);
    }
    else if (data.param != param)
    {
        data.param.copyValueFrom(&param);
        const DataType* vals = data.param.getVal<const DataType*>();
        ito::IntMeta *im;

        for (int i = 0; i < len; ++i)
        {
            if (metaNew)
            {
                im = new ito::IntMeta(metaNew->getMin(), metaNew->getMax(), metaNew->getStepSize(), metaNew->getCategory());
            }
            else
            {
                im = NULL;
            }
            ito::Param val("val", ito::ParamBase::Int, vals[i], im, "");
            d_ptr->m_numberPropertyManager->setParam(d_ptr->m_propertyToValues[property][i], val);
        }

        emit valueChanged(property, len, data.param.getVal<const DataType*>());
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
void ParamIntArrayPropertyManager::setValue(QtProperty *property, int num, const int* values)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = AbstractParamPropertyManager::d_ptr->m_values.find(property);
    if (it == AbstractParamPropertyManager::d_ptr->m_values.end())
        return;

    PrivateData &data = it.value();
    DataType* vals = data.param.getVal<DataType*>();
    int len = data.param.getLen();

    if (len != num || memcmp(values, vals, num * sizeof(DataType)) != 0)
    {
        data.param.setVal<const DataType*>(values, num);

        for (int i = 0; i < d_ptr->m_propertyToValues[property].size(); ++i)
        {
            d_ptr->m_numberPropertyManager->setValue(d_ptr->m_propertyToValues[property][i], values[i]);
        }

        emit valueChanged(property, num, values);
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
QString ParamIntArrayPropertyManager::valueText(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = AbstractParamPropertyManager::d_ptr->m_values.constFind(property);
    if (it == AbstractParamPropertyManager::d_ptr->m_values.constEnd())
        return QString();

    const ito::Param &param = it.value().param;
    int len = param.getLen();
    const DataType* vals = param.getVal<const DataType*>();

    switch (len)
    {
    case 0:
        return tr("<empty>");
    case 1:
        return QString("[%1]").arg(vals[0]);
    case 2:
        return QString("[%1,%2]").arg(vals[0]).arg(vals[1]);
    case 3:
        return QString("[%1,%2,%3]").arg(vals[0]).arg(vals[1]).arg(vals[2]);
    default:
        return tr("<%1 values>").arg(len);
    }
}

/*!
Returns the manager that creates the nested subproperties.

In order to provide editing widgets for the mentioned
subproperties in a property browser widget, this manager must be
associated with an editor factory.
*/
ParamIntPropertyManager *ParamIntArrayPropertyManager::subPropertyManager() const
{
    return d_ptr->m_numberPropertyManager;
}






//------------------------------------------------------------------------------
// ParamDoubleArrayPropertyManagerPrivate
class ParamDoubleArrayPropertyManagerPrivate
{
    ParamDoubleArrayPropertyManager *q_ptr;
    Q_DECLARE_PUBLIC(ParamDoubleArrayPropertyManager)
public:

    void slotValueChanged(QtProperty *property, double value);
    void slotPropertyDestroyed(QtProperty *property);
    void setMeta(QtProperty *property, const ito::DoubleArrayMeta &meta);

    ito::ParamDoublePropertyManager *m_numberPropertyManager;
    ito::AbstractParamPropertyManagerPrivate *m_d_ptr;

    QMap<const QtProperty*, QList<QtProperty*> > m_propertyToValues;
    QMap<const QtProperty*, QtProperty*> m_valuesToProperty;
};

void ParamDoubleArrayPropertyManagerPrivate::slotValueChanged(QtProperty *property, double value)
{
    if (QtProperty *prop = m_valuesToProperty.value(property, 0))
    {
        if (m_propertyToValues.contains(prop))
        {
            int index = m_propertyToValues[prop].indexOf(property);
            int num = m_d_ptr->m_values[prop].param.getLen();
            if (index >= 0 && index < num)
            {
                const ParamDoubleArrayPropertyManager::DataType* vals = m_d_ptr->m_values[prop].param.getVal<const ParamDoubleArrayPropertyManager::DataType*>();
                ParamDoubleArrayPropertyManager::DataType* newvals = (ParamDoubleArrayPropertyManager::DataType*)malloc(num * sizeof(ParamDoubleArrayPropertyManager::DataType));
                memcpy(newvals, vals, sizeof(ParamDoubleArrayPropertyManager::DataType) * num);
                newvals[index] = value;
                q_ptr->setValue(prop, num, newvals);
                free(newvals);
            }
        }
    }
}

void ParamDoubleArrayPropertyManagerPrivate::slotPropertyDestroyed(QtProperty *property)
{
    if (QtProperty *pointProp = m_valuesToProperty.value(property, 0))
    {
        m_propertyToValues[pointProp].removeOne(property);
        if (m_propertyToValues[pointProp].size() == 0)
        {
            m_propertyToValues.remove(pointProp);
            m_valuesToProperty.remove(property);
        }
    }
}

//------------------------------------------------------------------------------
ParamDoubleArrayPropertyManager::ParamDoubleArrayPropertyManager(QObject *parent /*= 0*/) :
AbstractParamPropertyManager(parent)
{
    d_ptr = new ParamDoubleArrayPropertyManagerPrivate;
    d_ptr->q_ptr = this;
    d_ptr->m_d_ptr = AbstractParamPropertyManager::d_ptr;

    d_ptr->m_numberPropertyManager = new ito::ParamDoublePropertyManager(this);
    connect(d_ptr->m_numberPropertyManager, SIGNAL(valueChanged(QtProperty *, double)),
        this, SLOT(slotValueChanged(QtProperty *, double)));
    connect(d_ptr->m_numberPropertyManager, SIGNAL(propertyDestroyed(QtProperty *)),
        this, SLOT(slotPropertyDestroyed(QtProperty *)));
}

//------------------------------------------------------------------------------
ParamDoubleArrayPropertyManager::~ParamDoubleArrayPropertyManager()
{
    delete d_ptr;
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
void ParamDoubleArrayPropertyManager::initializeProperty(QtProperty *property)
{
    DataType *ptr = NULL;
    AbstractParamPropertyManager::d_ptr->m_values[property] = AbstractParamPropertyManagerPrivate::Data(ito::Param("", ito::ParamBase::DoubleArray, 0, ptr, NULL, ""));
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
void ParamDoubleArrayPropertyManager::uninitializeProperty(QtProperty *property)
{
    foreach(QtProperty* prop, d_ptr->m_propertyToValues[property])
    {
        d_ptr->m_valuesToProperty.remove(prop);
        delete prop;
    }
    d_ptr->m_propertyToValues.remove(property);
}

//------------------------------------------------------------------------------
/*!
Sets the value of the given \a property to \a value.

If the specified \a value is not valid according to the given \a
property's range, the \a value is adjusted to the nearest valid
value within the range.

\sa value(), setRange(), valueChanged()
*/
void ParamDoubleArrayPropertyManager::setParam(QtProperty *property, const ito::Param &param)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = AbstractParamPropertyManager::d_ptr->m_values.find(property);
    if (it == AbstractParamPropertyManager::d_ptr->m_values.end())
        return;

    Q_ASSERT(param.getType() == (ito::ParamBase::DoubleArray));

    property->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));

    PrivateData &data = it.value();
    ito::DoubleArrayMeta *meta = data.param.getMetaT<ito::DoubleArrayMeta>();
    const ito::DoubleArrayMeta *metaNew = param.getMetaT<const ito::DoubleArrayMeta>();
    int len = std::max(0, param.getLen());
    const DataType* values = param.getVal<const DataType*>();

    int oldSubItems = d_ptr->m_propertyToValues.contains(property) ? d_ptr->m_propertyToValues[property].size() : 0;

    if (oldSubItems > len)
    {
        //remove supernumerous sub-items
        for (int i = oldSubItems; i = len; --i)
        {
            QtProperty *subProp = d_ptr->m_propertyToValues[property][i - 1];
            property->removeSubProperty(subProp);
            d_ptr->m_valuesToProperty.remove(subProp);
            delete subProp;
            d_ptr->m_propertyToValues[property].removeLast();
        }
    }

    if (len > oldSubItems)
    {
        //add new sub-items
        for (int i = oldSubItems; i < len; ++i)
        {
            QtProperty *prop = d_ptr->m_numberPropertyManager->addProperty();
            prop->setPropertyName(tr("%1").arg(i));
            d_ptr->m_numberPropertyManager->setValue(prop, 0);
            d_ptr->m_propertyToValues[property].append(prop);
            d_ptr->m_valuesToProperty[prop] = property;
            property->addSubProperty(prop);
        }
    }

    if ((meta && metaNew && (*meta != *metaNew)) || \
        (meta && !metaNew) || \
        (!meta && metaNew))
    {
        data.param = param;

        const DataType* vals = data.param.getVal<const DataType*>();
        ito::DoubleMeta *im;

        for (int i = 0; i < len; ++i)
        {
            im = new ito::DoubleMeta(metaNew->getMin(), metaNew->getMax(), metaNew->getStepSize(), metaNew->getCategory());
            ito::Param val("val", ito::ParamBase::Double, vals[i], im, "");
            d_ptr->m_numberPropertyManager->setParam(d_ptr->m_propertyToValues[property][i], val);
        }

        emit metaChanged(property, *metaNew);
        emit valueChanged(property, len, vals);
        emit propertyChanged(property);
    }
    else if (data.param != param)
    {
        data.param.copyValueFrom(&param);
        const DataType* vals = data.param.getVal<const DataType*>();
        ito::DoubleMeta *im;

        for (int i = 0; i < len; ++i)
        {
            if (metaNew)
            {
                im = new ito::DoubleMeta(metaNew->getMin(), metaNew->getMax(), metaNew->getStepSize(), metaNew->getCategory());
            }
            else
            {
                im = NULL;
            }
            ito::Param val("val", ito::ParamBase::Double, vals[i], im, "");
            d_ptr->m_numberPropertyManager->setParam(d_ptr->m_propertyToValues[property][i], val);
        }

        emit valueChanged(property, len, data.param.getVal<const DataType*>());
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
void ParamDoubleArrayPropertyManager::setValue(QtProperty *property, int num, const double* values)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = AbstractParamPropertyManager::d_ptr->m_values.find(property);
    if (it == AbstractParamPropertyManager::d_ptr->m_values.end())
        return;

    PrivateData &data = it.value();
    DataType* vals = data.param.getVal<DataType*>();
    int len = data.param.getLen();

    if (len != num || memcmp(values, vals, num * sizeof(DataType)) != 0)
    {
        data.param.setVal<const DataType*>(values, num);

        for (int i = 0; i < d_ptr->m_propertyToValues[property].size(); ++i)
        {
            d_ptr->m_numberPropertyManager->setValue(d_ptr->m_propertyToValues[property][i], values[i]);
        }

        emit valueChanged(property, num, values);
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
QString ParamDoubleArrayPropertyManager::valueText(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = AbstractParamPropertyManager::d_ptr->m_values.constFind(property);
    if (it == AbstractParamPropertyManager::d_ptr->m_values.constEnd())
        return QString();

    const ito::Param &param = it.value().param;
    int len = param.getLen();
    const DataType* vals = param.getVal<const DataType*>();
    QLocale locale;

    switch (len)
    {
    case 0:
        return tr("<empty>");
    case 1:
        return QString("[%1]").arg(locale.toString(vals[0]));
    case 2:
        return QString("[%1,%2]").arg(locale.toString(vals[0]), locale.toString(vals[1]));
    case 3:
        return QString("[%1,%2,%3]").arg(locale.toString(vals[0]), locale.toString(vals[1]), locale.toString(vals[2]));
    default:
        return tr("<%1 values>").arg(len);
    }
}

/*!
Returns the manager that creates the nested subproperties.

In order to provide editing widgets for the mentioned
subproperties in a property browser widget, this manager must be
associated with an editor factory.
*/
ParamDoublePropertyManager *ParamDoubleArrayPropertyManager::subPropertyManager() const
{
    return d_ptr->m_numberPropertyManager;
}





//------------------------------------------------------------------------------
// ParamStringListPropertyManagerPrivate
class ParamStringListPropertyManagerPrivate
{
    ParamStringListPropertyManager *q_ptr;
    Q_DECLARE_PUBLIC(ParamStringListPropertyManager)
public:

    void slotValueChanged(QtProperty *property, const QByteArray &value);
    void slotPropertyDestroyed(QtProperty *property);
    void setMeta(QtProperty *property, const ito::StringListMeta &meta);

    ito::ParamStringPropertyManager *m_singleItemPropertyManager;
    ito::AbstractParamPropertyManagerPrivate *m_d_ptr;

    QMap<const QtProperty*, QList<QtProperty*> > m_propertyToValues;
    QMap<const QtProperty*, QtProperty*> m_valuesToProperty;
};

void ParamStringListPropertyManagerPrivate::slotValueChanged(QtProperty *property, const QByteArray &value)
{
    if (QtProperty *prop = m_valuesToProperty.value(property, 0))
    {
        if (m_propertyToValues.contains(prop))
        {
            int index = m_propertyToValues[prop].indexOf(property);
            int num = m_d_ptr->m_values[prop].param.getLen();
            if (index >= 0 && index < num)
            {
                const ParamStringListPropertyManager::DataType* vals = m_d_ptr->m_values[prop].param.getVal<const ParamStringListPropertyManager::DataType*>();
                ParamStringListPropertyManager::DataType* newvals = new ParamStringListPropertyManager::DataType[num];

                for (int i = 0; i < num; ++i)
                {
                    newvals[i] = vals[i];
                }

                newvals[index] = value.constData();
                q_ptr->setValue(prop, num, newvals);
                delete[] newvals;
            }
        }
    }
}

void ParamStringListPropertyManagerPrivate::slotPropertyDestroyed(QtProperty *property)
{
    if (QtProperty *pointProp = m_valuesToProperty.value(property, 0))
    {
        m_propertyToValues[pointProp].removeOne(property);
        if (m_propertyToValues[pointProp].size() == 0)
        {
            m_propertyToValues.remove(pointProp);
            m_valuesToProperty.remove(property);
        }
    }
}

//------------------------------------------------------------------------------
ParamStringListPropertyManager::ParamStringListPropertyManager(QObject *parent /*= 0*/) :
    AbstractParamPropertyManager(parent)
{
    d_ptr = new ParamStringListPropertyManagerPrivate;
    d_ptr->q_ptr = this;
    d_ptr->m_d_ptr = AbstractParamPropertyManager::d_ptr;

    d_ptr->m_singleItemPropertyManager = new ito::ParamStringPropertyManager(this);
    connect(d_ptr->m_singleItemPropertyManager, SIGNAL(valueChanged(QtProperty *, QByteArray)),
        this, SLOT(slotValueChanged(QtProperty *, QByteArray)));
    connect(d_ptr->m_singleItemPropertyManager, SIGNAL(propertyDestroyed(QtProperty *)),
        this, SLOT(slotPropertyDestroyed(QtProperty *)));
}

//------------------------------------------------------------------------------
ParamStringListPropertyManager::~ParamStringListPropertyManager()
{
    delete d_ptr;
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
void ParamStringListPropertyManager::initializeProperty(QtProperty *property)
{
    DataType *ptr = nullptr;
    AbstractParamPropertyManager::d_ptr->m_values[property] = AbstractParamPropertyManagerPrivate::Data(ito::Param("", ito::ParamBase::StringList, 0, ptr, nullptr, ""));
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
void ParamStringListPropertyManager::uninitializeProperty(QtProperty *property)
{
    foreach(QtProperty* prop, d_ptr->m_propertyToValues[property])
    {
        d_ptr->m_valuesToProperty.remove(prop);
        delete prop;
    }
    d_ptr->m_propertyToValues.remove(property);
}

//------------------------------------------------------------------------------
/*!
Sets the value of the given \a property to \a value.

If the specified \a value is not valid according to the given \a
property's range, the \a value is adjusted to the nearest valid
value within the range.

\sa value(), setRange(), valueChanged()
*/
void ParamStringListPropertyManager::setParam(QtProperty *property, const ito::Param &param)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = AbstractParamPropertyManager::d_ptr->m_values.find(property);
    if (it == AbstractParamPropertyManager::d_ptr->m_values.end())
        return;

    Q_ASSERT(param.getType() == (ito::ParamBase::StringList));

    property->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));

    PrivateData &data = it.value();
    auto meta = data.param.getMetaT<ito::StringListMeta>();
    const auto metaNew = param.getMetaT<const ito::StringListMeta>();
    int len = std::max(0, param.getLen());
    const auto values = param.getVal<const DataType*>();

    int oldSubItems = d_ptr->m_propertyToValues.contains(property) ? d_ptr->m_propertyToValues[property].size() : 0;

    if (oldSubItems > len)
    {
        //remove supernumerous sub-items
        for (int i = oldSubItems; i = len; --i)
        {
            QtProperty *subProp = d_ptr->m_propertyToValues[property][i - 1];
            property->removeSubProperty(subProp);
            d_ptr->m_valuesToProperty.remove(subProp);
            delete subProp;
            d_ptr->m_propertyToValues[property].removeLast();
        }
    }

    if (len > oldSubItems)
    {
        //add new sub-items
        for (int i = oldSubItems; i < len; ++i)
        {
            QtProperty *prop = d_ptr->m_singleItemPropertyManager->addProperty();
            prop->setPropertyName(tr("%1").arg(i));
            d_ptr->m_singleItemPropertyManager->setValue(prop, "");
            d_ptr->m_propertyToValues[property].append(prop);
            d_ptr->m_valuesToProperty[prop] = property;
            property->addSubProperty(prop);
        }
    }

    if ((meta && metaNew && (*meta != *metaNew)) || \
        (meta && !metaNew) || \
        (!meta && metaNew))
    {
        data.param = param;

        const auto vals = data.param.getVal<const DataType*>();
        ito::StringMeta *im;

        for (int i = 0; i < len; ++i)
        {
            im = new ito::StringMeta(*metaNew);
            ito::Param val("val", ito::ParamBase::String, vals[i].data(), "");
            val.setMeta(im, true);
            d_ptr->m_singleItemPropertyManager->setParam(d_ptr->m_propertyToValues[property][i], val);
        }

        emit metaChanged(property, *metaNew);
        emit valueChanged(property, len, vals);
        emit propertyChanged(property);
    }
    else if (data.param != param)
    {
        data.param.copyValueFrom(&param);
        const auto vals = data.param.getVal<const DataType*>();
        ito::StringMeta *im;

        for (int i = 0; i < len; ++i)
        {
            if (metaNew)
            {
                im = new ito::StringMeta(*metaNew);
            }
            else
            {
                im = NULL;
            }
            ito::Param val("val", ito::ParamBase::String, vals[i].data(), "");
            val.setMeta(im, true);
            d_ptr->m_singleItemPropertyManager->setParam(d_ptr->m_propertyToValues[property][i], val);
        }

        emit valueChanged(property, len, data.param.getVal<const DataType*>());
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
void ParamStringListPropertyManager::setValue(QtProperty *property, int num, const ito::ByteArray* values)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = AbstractParamPropertyManager::d_ptr->m_values.find(property);
    if (it == AbstractParamPropertyManager::d_ptr->m_values.end())
        return;

    PrivateData &data = it.value();
    auto vals = data.param.getVal<DataType*>();
    int len = data.param.getLen();
    bool changed = len != num;

    if (!changed)
    {
        for (int i = 0; i < num; ++i)
        {
            changed |= (values[i] != vals[i]);
        }
    }

    if (changed)
    {
        data.param.setVal<const DataType*>(values, num);

        for (int i = 0; i < d_ptr->m_propertyToValues[property].size(); ++i)
        {
            d_ptr->m_singleItemPropertyManager->setValue(d_ptr->m_propertyToValues[property][i], values[i].data());
        }

        emit valueChanged(property, num, values);
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
QString ParamStringListPropertyManager::valueText(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = AbstractParamPropertyManager::d_ptr->m_values.constFind(property);
    if (it == AbstractParamPropertyManager::d_ptr->m_values.constEnd())
        return QString();

    const ito::Param &param = it.value().param;
    int len = param.getLen();
    const auto vals = param.getVal<const DataType*>();

    switch (len)
    {
    case 0:
        return tr("<empty>");
    case 1:
        return QString("[%1]").arg(QLatin1String(vals[0].data()));
    case 2:
        return QString("[%1,%2]").arg(QLatin1String(vals[0].data())).arg(QLatin1String(vals[1].data()));
    case 3:
        return QString("[%1,%2,%3]").arg(QLatin1String(vals[0].data())).arg(QLatin1String(vals[1].data())).arg(QLatin1String(vals[2].data()));
    default:
        return tr("<%1 values>").arg(len);
    }
}

/*!
Returns the manager that creates the nested subproperties.

In order to provide editing widgets for the mentioned
subproperties in a property browser widget, this manager must be
associated with an editor factory.
*/
ParamStringPropertyManager *ParamStringListPropertyManager::subPropertyManager() const
{
    return d_ptr->m_singleItemPropertyManager;
}









//------------------------------------------------------------------------------
ParamOtherPropertyManager::ParamOtherPropertyManager(QObject *parent /*= 0*/) :
AbstractParamPropertyManager(parent)
{
}

//------------------------------------------------------------------------------
ParamOtherPropertyManager::~ParamOtherPropertyManager()
{
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
void ParamOtherPropertyManager::initializeProperty(QtProperty *property)
{
    d_ptr->m_values[property] = AbstractParamPropertyManagerPrivate::Data(ito::Param());
}

//------------------------------------------------------------------------------
/*!
\fn void AbstractParamPropertyManager::setValue(QtProperty *property, int value)

Sets the value of the given \a property to \a value.

If the specified \a value is not valid according to the given \a
property's range, the \a value is adjusted to the nearest valid
value within the range.

\sa value(), setRange(), valueChanged()
*/
void ParamOtherPropertyManager::setParam(QtProperty *property, const ito::Param &param)
{
    typedef AbstractParamPropertyManagerPrivate::Data PrivateData;
    typedef QMap<const QtProperty *, PrivateData> PropertyToData;
    typedef PropertyToData::iterator PropertyToDataIterator;

    const PropertyToDataIterator it = d_ptr->m_values.find(property);
    if (it == d_ptr->m_values.end())
        return;

    Q_ASSERT((param.getType() == (ito::ParamBase::HWRef)) |
        (param.getType() == (ito::ParamBase::DObjPtr)) |
        (param.getType() == (ito::ParamBase::PointPtr)) |
        (param.getType() == (ito::ParamBase::PointCloudPtr)) |
        (param.getType() == (ito::ParamBase::PolygonMeshPtr)) |
        (param.getType() == (ito::ParamBase::ComplexArray)) |
        (param.getType() == ito::ParamBase::Complex));

    property->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));

    PrivateData &data = it.value();
    if (data.param != param)
    {
        if (data.param.getType() == param.getType())
        {
            data.param.copyValueFrom(&param);
        }
        else
        {
            data.param = param;
        }
        emit propertyChanged(property);
    }
}

//------------------------------------------------------------------------------
/*!
\reimp
*/
QString ParamOtherPropertyManager::valueText(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = d_ptr->m_values.constFind(property);
    if (it == d_ptr->m_values.constEnd())
        return QString();

    const ito::Param &param = it.value().param;

    switch (param.getType())
    {
    case (ito::ParamBase::HWRef):
        {
            ito::AddInBase *aib = param.getVal<ito::AddInBase*>();
            if (aib)
            {
                return QString("%1: %2").arg(aib->objectName()).arg(aib->getIdentifier());
            }
            else
            {
                return QLatin1String("None");
            }
        }
        break;
    case (ito::ParamBase::DObjPtr):
        {
            ito::DataObject *obj = param.getVal<ito::DataObject*>();
            if (obj)
            {
                return QString("DataObject");
            }
            else
            {
                return QLatin1String("None");
            }
        }
        break;
    case (ito::ParamBase::PointPtr):
        {
            return QLatin1String("Point");
        }
        break;
    case (ito::ParamBase::PolygonMeshPtr):
        {
            return QLatin1String("PolygonMesh");
        }
        break;
    case (ito::ParamBase::PointCloudPtr):
        {
            return QLatin1String("PointCloud");
        }
        break;
    case (ito::ParamBase::Complex):
        {
            auto val = param.getVal<ito::complex128>();
            QString s;
            QLocale locale;

            if (val.imag() >= 0)
            {
                s = locale.toString(val.real(), 'g', 4) + "+" + locale.toString(val.imag(), 'g', 4) + "i";
            }
            else
            {
                s = locale.toString(val.real(), 'g', 4) + "-" + locale.toString(std::abs(val.imag()), 'g', 4) + "i";
            }

            return s;
            break;
        }
    case (ito::ParamBase::ComplexArray):
        return tr("%1 complex values").arg(std::max(0, param.getLen()));
        break;
    }

    return QLatin1String("Unknown");
}



} //end namespace ito


#include "moc_itomParamManager.cpp"
#include "itomParamManager.moc"
