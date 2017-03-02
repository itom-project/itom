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
        Data() : param("", ito::ParamBase::Int, 0, NULL, NULL) {}
        ito::Param param;
    };

    typedef QMap<const QtProperty *, Data> PropertyValueMap;
    PropertyValueMap m_values;
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
        return ito::ParamBase();
    return it.value().param;
}

//------------------------------------------------------------------------------
const ito::Param &AbstractParamPropertyManager::param(const QtProperty *property) const
{
    const AbstractParamPropertyManagerPrivate::PropertyValueMap::const_iterator it = d_ptr->m_values.constFind(property);
    if (it == d_ptr->m_values.constEnd())
        return ito::Param();
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
void AbstractParamPropertyManager::initializeProperty(QtProperty *property)
{
    d_ptr->m_values[property] = AbstractParamPropertyManagerPrivate::Data();
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
        data.param.setVal<int>(param.getVal<int>());
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
        if (meta->getMin() == 0 && meta->getMax() && meta->getStepSize() == 1)
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
    if (meta && meta->getMin() == 0 && meta->getMax() == 1 && meta->getStepSize() == 1)
    {
        static const QIcon checkedIcon = drawCheckBox(true);
        static const QIcon uncheckedIcon = drawCheckBox(false);
        return param.getVal<int>() > 0 ? checkedIcon : uncheckedIcon;
    }

    return QIcon();
}



} //end namespace ito


#include "moc_itomParamManager.cpp"
#include "itomParamManager.moc"