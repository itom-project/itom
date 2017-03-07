/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2017, Institut fuer Technische Optik (ITO),
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
void ParamIntervalPropertyManager::setParam(QtProperty *property, const ito::Param &param)
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
    return QLatin1String(param.getVal<const char*>());
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
    widthProp->setPropertyName(tr("Width"));
    d_ptr->m_intervalPropertyManager->setValue(widthProp, 0, 0);
    d_ptr->m_propertyToWidth[property] = widthProp;
    d_ptr->m_widthToProperty[widthProp] = property;
    property->addSubProperty(widthProp);

    QtProperty *heightProp = d_ptr->m_intervalPropertyManager->addProperty();
    heightProp->setPropertyName(tr("Top"));
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

    Q_ASSERT(param.getType() == ito::ParamBase::String);

    property->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));

    PrivateData &data = it.value();
    ito::RectMeta *meta = data.param.getMetaT<ito::RectMeta>();
    const ito::RectMeta *metaNew = param.getMetaT<const ito::RectMeta>();

    if ((meta && metaNew && (*meta != *metaNew)) || \
        (meta && !metaNew) || \
        (!meta && metaNew))
    {
        const int* vals = data.param.getVal<const int*>();
        data.param = param;
        int vw[] = {vals[0], vals[0] + vals[2] - 1};
        ito::Param width("width", ito::ParamBase::IntArray, 2, vw, new ito::RangeMeta(metaNew->getWidthRangeMeta()), "");
        d_ptr->m_intervalPropertyManager->setParam(d_ptr->m_propertyToHeight[property], width);
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
    return QString("%1, %2, %3, %4").arg(vals[0]).arg(vals[1]).arg(vals[2]).arg(vals[3]);
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

    Q_ASSERT((param.getType() == (ito::ParamBase::HWRef & ito::paramTypeMask)) |
        (param.getType() == (ito::ParamBase::DObjPtr & ito::paramTypeMask)) |
        (param.getType() == (ito::ParamBase::PointPtr & ito::paramTypeMask)) |
        (param.getType() == (ito::ParamBase::PointCloudPtr & ito::paramTypeMask)) |
        (param.getType() == (ito::ParamBase::PolygonMeshPtr & ito::paramTypeMask)));

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
    case (ito::ParamBase::HWRef & ito::paramTypeMask):
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
    case (ito::ParamBase::DObjPtr & ito::paramTypeMask):
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
    case (ito::ParamBase::PointPtr & ito::paramTypeMask):
        {
            return QLatin1String("Point");
        }
        break;
    case (ito::ParamBase::PolygonMeshPtr & ito::paramTypeMask):
        {
            return QLatin1String("PolygonMesh");
        }
        break;
    case (ito::ParamBase::PointCloudPtr & ito::paramTypeMask):
        {
            return QLatin1String("PointCloud");
        }
        break;
    }

    return QLatin1String("Unknown");
}



} //end namespace ito


#include "moc_itomParamManager.cpp"
#include "itomParamManager.moc"