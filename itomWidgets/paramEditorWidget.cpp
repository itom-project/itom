/* ********************************************************************
itom measurement system
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2017, Institut fuer Technische Optik (ITO),
Universitaet Stuttgart, Germany

This file is part of itom.

itom is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

itom is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#include "paramEditorWidget.h"
#include <qevent.h>
#include <qheaderview.h>
#include <qaction.h>
#include <qlayout.h>
#include <qmessagebox.h>
#include <qtimer.h>

#include "common/addInInterface.h"

#include "qttreepropertybrowser.h"
#include "qtgroupboxpropertybrowser.h"
#include "itomParamManager.h"
#include "qtpropertymanager.h"
#include "itomParamFactory.h"

class ParamEditorWidgetPrivate
{
	//Q_DECLARE_PUBLIC(ParamEditorWidget);

public:
    ParamEditorWidgetPrivate()
        : m_pBrowser(NULL),
        m_pIntManager(NULL),
        m_pCharManager(NULL),
        m_pDoubleManager(NULL),
        m_pStringManager(NULL),
        m_pOtherManager(NULL),
        m_pIntervalManager(NULL),
        m_pRectManager(NULL),
        m_timerID(-1),
        m_isChanging(false),
        m_readonly(false)
    {};

    void clearGroups()
    {
        QMap<QString, QtProperty*>::iterator it = m_groups.begin();
        while (it != m_groups.end())
        {
            delete it.value();
            ++it;
        }
        m_groups.clear();
    }

    void enqueue(const QSharedPointer<ito::ParamBase> param)
    {
        const char* name = param->getName();

        for (int i = 0; i < m_changedParameters.size(); ++i)
        {
            if (memcmp(m_changedParameters[i]->getName(), name, sizeof(name)) == 0)
            {
                m_changedParameters[i] = param;
                return;
            }
        }

        m_changedParameters.append(param);
    }

    QPointer<ito::AddInBase> m_plugin;
    QtTreePropertyBrowser *m_pBrowser;

    ito::ParamIntPropertyManager *m_pIntManager;
    ito::ParamCharPropertyManager *m_pCharManager;
    ito::ParamDoublePropertyManager *m_pDoubleManager;
    ito::ParamStringPropertyManager *m_pStringManager;
    ito::ParamOtherPropertyManager *m_pOtherManager;
    ito::ParamIntervalPropertyManager *m_pIntervalManager;
    ito::ParamRectPropertyManager *m_pRectManager;
    QtGroupPropertyManager *m_pGroupPropertyManager;

    //factories, responsible for editing properties.
    ito::ParamIntPropertyFactory *m_pIntFactory;
    ito::ParamStringPropertyFactory *m_pStringFactory;
    ito::ParamIntervalPropertyFactory *m_pIntervalFactory;

    QVector<QSharedPointer<ito::ParamBase> > m_changedParameters;
    QMap<QByteArray, QtProperty*> m_properties;
    QMap<QString, QtProperty*> m_groups;
    int m_timerID;
    bool m_isChanging;
    bool m_readonly;
};

//-----------------------------------------------------------------------
ParamEditorWidget::ParamEditorWidget(QWidget* parent /*= 0*/) : 
	QWidget(parent),
	d_ptr(new ParamEditorWidgetPrivate())
{
	Q_D(ParamEditorWidget);

	d->m_pBrowser = new QtTreePropertyBrowser();

    d->m_pGroupPropertyManager = new QtGroupPropertyManager(this);

    d->m_pIntManager = new ito::ParamIntPropertyManager(this);
    connect(d->m_pIntManager, SIGNAL(valueChanged(QtProperty *, int)), this, SLOT(valueChanged(QtProperty *, int)));
    d->m_pIntFactory = new ito::ParamIntPropertyFactory(this);

    d->m_pCharManager = new ito::ParamCharPropertyManager(this);
    connect(d->m_pCharManager, SIGNAL(valueChanged(QtProperty *, char)), this, SLOT(valueChanged(QtProperty *, char)));
    //d->m_pCharFactory = new ito::ParamCharPropertyFactory(this);

    d->m_pDoubleManager = new ito::ParamDoublePropertyManager(this);
    connect(d->m_pDoubleManager, SIGNAL(valueChanged(QtProperty *, double)), this, SLOT(valueChanged(QtProperty *, double)));
    //d->m_pDoubleFactory = new ito::ParamDoublePropertyFactory(this);

    d->m_pStringManager = new ito::ParamStringPropertyManager(this);
    connect(d->m_pStringManager, SIGNAL(valueChanged(QtProperty *, QByteArray)), this, SLOT(valueChanged(QtProperty *, QByteArray)));
    d->m_pStringFactory = new ito::ParamStringPropertyFactory(this);

    d->m_pIntervalManager = new ito::ParamIntervalPropertyManager(this);
    connect(d->m_pIntervalManager, SIGNAL(valueChanged(QtProperty *, int, int)), this, SLOT(valueChanged(QtProperty *, int, int)));
    d->m_pIntervalFactory = new ito::ParamIntervalPropertyFactory(this);

    d->m_pRectManager = new ito::ParamRectPropertyManager(this); 
    connect(d->m_pRectManager, SIGNAL(valueChanged(QtProperty *, int, int, int, int)), this, SLOT(valueChanged(QtProperty *, int, int, int, int)));

    d->m_pOtherManager = new ito::ParamOtherPropertyManager(this);    

    //the following command set all factories for the managers
    d_ptr->m_readonly = true;
    setReadonly(false);

    QHBoxLayout *hboxLayout = new QHBoxLayout();
    hboxLayout->setMargin(0);
    hboxLayout->addWidget(d->m_pBrowser);
    setLayout(hboxLayout);

}

//-----------------------------------------------------------------------
ParamEditorWidget::~ParamEditorWidget()
{
    Q_D(ParamEditorWidget);
    DELETE_AND_SET_NULL(d->m_pIntFactory);
    DELETE_AND_SET_NULL(d->m_pStringFactory);
    DELETE_AND_SET_NULL(d->m_pIntervalFactory);
    DELETE_AND_SET_NULL(d->m_pIntManager);
    DELETE_AND_SET_NULL(d->m_pCharManager);
    DELETE_AND_SET_NULL(d->m_pDoubleManager);
    DELETE_AND_SET_NULL(d->m_pStringManager);
    DELETE_AND_SET_NULL(d->m_pIntervalManager);
    DELETE_AND_SET_NULL(d->m_pRectManager);
    DELETE_AND_SET_NULL(d->m_pOtherManager);
    DELETE_AND_SET_NULL(d->m_pGroupPropertyManager);
    DELETE_AND_SET_NULL(d->m_pBrowser);
}

//-----------------------------------------------------------------------
QPointer<ito::AddInBase> ParamEditorWidget::plugin() const
{
    Q_D(const ParamEditorWidget);
    return d->m_plugin;
}

//-----------------------------------------------------------------------
void ParamEditorWidget::setPlugin(QPointer<ito::AddInBase> plugin)
{
    Q_D(ParamEditorWidget);

    if (d->m_plugin.data() != plugin.data())
    {
        if (d->m_plugin)
        {
            disconnect(this, SLOT(parametersChanged(QMap<QString, ito::Param>)));
        }

        d->m_plugin = plugin;

        d->m_pBrowser->clear();
        d->clearGroups();

        if (plugin.isNull() == false)
        {
            if (d->m_plugin)
            {
                connect(d->m_plugin.data(), SIGNAL(parametersChanged(QMap<QString, ito::Param>)), this, SLOT(parametersChanged(QMap<QString, ito::Param>)));
            }

            d->m_groups["General"] = d->m_pGroupPropertyManager->addProperty("General");
            d->m_pBrowser->addProperty(d->m_groups["General"]);

            QMap<QString, ito::Param> *params;
            QMap<QString, ito::Param>::const_iterator iter;
            plugin->getParamList(&params);

            iter = params->constBegin();
            while (iter != params->constEnd())
            {
                addParam(*iter);
                ++iter;
            }
            
        }
    }
}

//-----------------------------------------------------------------------
bool ParamEditorWidget::readonly() const
{
    Q_D(const ParamEditorWidget);
    return d_ptr->m_readonly;
}

//-----------------------------------------------------------------------
void ParamEditorWidget::setReadonly(bool enable)
{
    Q_D(ParamEditorWidget);
    if (enable != d_ptr->m_readonly)
    {
        d_ptr->m_readonly = enable;

        if (enable)
        {
            d->m_pBrowser->unsetFactoryForManager(d->m_pIntManager);
            d->m_pBrowser->unsetFactoryForManager(d->m_pCharManager);
            d->m_pBrowser->unsetFactoryForManager(d->m_pDoubleManager);
            d->m_pBrowser->unsetFactoryForManager(d->m_pStringManager);
            d->m_pBrowser->unsetFactoryForManager(d->m_pIntervalManager);
            d->m_pBrowser->unsetFactoryForManager(d->m_pRectManager->subIntervalPropertyManager());
        }
        else
        {
            d->m_pBrowser->setFactoryForManager(d->m_pIntManager, d->m_pIntFactory);
            //d->m_pBrowser->setFactoryForManager(d->m_pIntManager, d->m_pCharFactory);
            //d->m_pBrowser->setFactoryForManager(d->m_pIntManager, d->m_pDoubleFactory);
            d->m_pBrowser->setFactoryForManager(d->m_pStringManager, d->m_pStringFactory);
            d->m_pBrowser->setFactoryForManager(d->m_pIntervalManager, d->m_pIntervalFactory);
            d->m_pBrowser->setFactoryForManager(d->m_pRectManager->subIntervalPropertyManager(), d->m_pIntervalFactory);
        }
    }
}

//-----------------------------------------------------------------------
int ParamEditorWidget::indentation() const
{
    Q_D(const ParamEditorWidget);
    return d_ptr->m_pBrowser->indentation();
}

//-----------------------------------------------------------------------
void ParamEditorWidget::setIndentation(int i)
{
    Q_D(ParamEditorWidget);
    d_ptr->m_pBrowser->setIndentation(i);
}

//-----------------------------------------------------------------------
bool ParamEditorWidget::rootIsDecorated() const
{
    Q_D(const ParamEditorWidget);
    return d_ptr->m_pBrowser->rootIsDecorated();
}

//-----------------------------------------------------------------------
void ParamEditorWidget::setRootIsDecorated(bool show)
{
    Q_D(ParamEditorWidget);
    d_ptr->m_pBrowser->setRootIsDecorated(show);
}

//-----------------------------------------------------------------------
bool ParamEditorWidget::alternatingRowColors() const
{
    Q_D(const ParamEditorWidget);
    return d_ptr->m_pBrowser->alternatingRowColors();
}

//-----------------------------------------------------------------------
void ParamEditorWidget::setAlternatingRowColors(bool enable)
{
    Q_D(ParamEditorWidget);
    d_ptr->m_pBrowser->setAlternatingRowColors(enable);
}

//-----------------------------------------------------------------------
bool ParamEditorWidget::isHeaderVisible() const
{
    Q_D(const ParamEditorWidget);
    return d_ptr->m_pBrowser->isHeaderVisible();
}

//-----------------------------------------------------------------------
void ParamEditorWidget::setHeaderVisible(bool visible)
{
    Q_D(ParamEditorWidget);
    d_ptr->m_pBrowser->setHeaderVisible(visible);
}

//-----------------------------------------------------------------------
ParamEditorWidget::ResizeMode ParamEditorWidget::resizeMode() const
{
    Q_D(const ParamEditorWidget);
    return (ParamEditorWidget::ResizeMode)d_ptr->m_pBrowser->resizeMode();
}

//-----------------------------------------------------------------------
void ParamEditorWidget::setResizeMode(ResizeMode mode)
{
    Q_D(ParamEditorWidget);
    d_ptr->m_pBrowser->setResizeMode((QtTreePropertyBrowser::ResizeMode)mode);
}

//-----------------------------------------------------------------------
int ParamEditorWidget::splitterPosition() const
{
    Q_D(const ParamEditorWidget);
    return d_ptr->m_pBrowser->splitterPosition();
}

//-----------------------------------------------------------------------
void ParamEditorWidget::setSplitterPosition(int position)
{
    Q_D(ParamEditorWidget);
    d_ptr->m_pBrowser->setSplitterPosition(position);
}

//-----------------------------------------------------------------------
void ParamEditorWidget::setPropertiesWithoutValueMarked(bool mark)
{
    Q_D(ParamEditorWidget);
    d_ptr->m_pBrowser->setPropertiesWithoutValueMarked(mark);
}

//-----------------------------------------------------------------------
bool ParamEditorWidget::propertiesWithoutValueMarked() const
{
    Q_D(const ParamEditorWidget);
    return d_ptr->m_pBrowser->propertiesWithoutValueMarked();
}

//-----------------------------------------------------------------------
ito::RetVal ParamEditorWidget::addParam(const ito::Param &param)
{
    Q_D(ParamEditorWidget);

    const ito::ParamMeta *meta = param.getMetaT<ito::ParamMeta>();
    QtProperty* groupProperty = NULL;
    ito::RetVal retval;
    ito::ParamMeta::MetaRtti metaType = meta ? meta->getType() : ito::ParamMeta::rttiUnknown;

    QString group = "General";
    if (meta && meta->getCategory().empty() == false)
    {
        group = meta->getCategory().data();
    }

    if (d->m_groups.contains(group))
    {
        groupProperty = d->m_groups[group];
    }
    else
    {
        d->m_groups[group] = d->m_pGroupPropertyManager->addProperty(group);
        d->m_pBrowser->addProperty(d->m_groups[group]);
        groupProperty = d->m_groups[group];
    }

    switch (param.getType())
    {
    case ito::ParamBase::Int:
        retval += addParamInt(param, groupProperty);
        break;
    case ito::ParamBase::Char:
        retval += addParamChar(param, groupProperty);
        break;
    case ito::ParamBase::Double:
        retval += addParamDouble(param, groupProperty);
        break;
    case ito::ParamBase::String:
        retval += addParamString(param, groupProperty);
        break;
    case ito::ParamBase::IntArray & ito::paramTypeMask:
        if ((metaType == ito::ParamMeta::rttiIntervalMeta) || (metaType == ito::ParamMeta::rttiRangeMeta))
        {
            retval += addParamInterval(param, groupProperty);
        }
        else if (metaType == ito::ParamMeta::rttiRectMeta)
        {
            retval += addParamRect(param, groupProperty);
        }
        break;
    case (ito::ParamBase::HWRef & ito::paramTypeMask):
    case (ito::ParamBase::DObjPtr & ito::paramTypeMask):
    case (ito::ParamBase::PointPtr & ito::paramTypeMask):
    case (ito::ParamBase::PolygonMeshPtr & ito::paramTypeMask):
    case (ito::ParamBase::PointCloudPtr & ito::paramTypeMask):
        retval += addParamOthers(param, groupProperty);
        break;
    default:
        retval += ito::RetVal::format(ito::retError, 0, "unsupported type of parameter '%s'", param.getName());
        break;
    }

    return retval;
}


//-----------------------------------------------------------------------
ito::RetVal ParamEditorWidget::addParamInt(const ito::Param &param, QtProperty *groupProperty)
{
    Q_D(ParamEditorWidget);

    d->m_pIntManager->blockSignals(true);
    QtProperty *prop = d->m_pIntManager->addProperty(param.getName());
    d->m_properties[param.getName()] = prop;
    prop->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));
    d->m_pIntManager->setParam(prop, param);
    if (groupProperty)
    {
        groupProperty->addSubProperty(prop);
    }
    else
    {
        d->m_pBrowser->addProperty(prop);
    }
    d->m_pIntManager->blockSignals(false);
    prop->setStatusTip(param.getInfo());
    prop->setToolTip(param.getInfo());

    return ito::retOk;
}

//-----------------------------------------------------------------------
ito::RetVal ParamEditorWidget::addParamChar(const ito::Param &param, QtProperty *groupProperty)
{
    Q_D(ParamEditorWidget);

    d->m_pCharManager->blockSignals(true);
    QtProperty *prop = d->m_pCharManager->addProperty(param.getName());
    d->m_properties[param.getName()] = prop;
    prop->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));
    d->m_pCharManager->setParam(prop, param);
    if (groupProperty)
    {
        groupProperty->addSubProperty(prop);
    }
    else
    {
        d->m_pBrowser->addProperty(prop);
    }
    d->m_pCharManager->blockSignals(false);
    prop->setStatusTip(param.getInfo());
    prop->setToolTip(param.getInfo());

    return ito::retOk;
}

//-----------------------------------------------------------------------
ito::RetVal ParamEditorWidget::addParamDouble(const ito::Param &param, QtProperty *groupProperty)
{
    Q_D(ParamEditorWidget);

    d->m_pDoubleManager->blockSignals(true);
    QtProperty *prop = d->m_pDoubleManager->addProperty(param.getName());
    d->m_properties[param.getName()] = prop;
    prop->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));
    d->m_pDoubleManager->setParam(prop, param);
    if (groupProperty)
    {
        groupProperty->addSubProperty(prop);
    }
    else
    {
        d->m_pBrowser->addProperty(prop);
    }
    d->m_pDoubleManager->blockSignals(false);
    prop->setStatusTip(param.getInfo());
    prop->setToolTip(param.getInfo());

    return ito::retOk;
}

//-----------------------------------------------------------------------
ito::RetVal ParamEditorWidget::addParamString(const ito::Param &param, QtProperty *groupProperty)
{
    Q_D(ParamEditorWidget);

    d->m_pStringManager->blockSignals(true);
    QtProperty *prop = d->m_pStringManager->addProperty(param.getName());
    d->m_properties[param.getName()] = prop;
    prop->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));
    d->m_pStringManager->setParam(prop, param);
    if (groupProperty)
    {
        groupProperty->addSubProperty(prop);
    }
    else
    {
        d->m_pBrowser->addProperty(prop);
    }
    d->m_pStringManager->blockSignals(false);
    prop->setStatusTip(param.getInfo());
    prop->setToolTip(param.getInfo());

    return ito::retOk;
}

//-----------------------------------------------------------------------
ito::RetVal ParamEditorWidget::addParamInterval(const ito::Param &param, QtProperty *groupProperty)
{
    Q_D(ParamEditorWidget);

    d->m_pIntervalManager->blockSignals(true);
    QtProperty *prop = d->m_pIntervalManager->addProperty(param.getName());
    d->m_properties[param.getName()] = prop;
    prop->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));
    d->m_pIntervalManager->setParam(prop, param);
    if (groupProperty)
    {
        groupProperty->addSubProperty(prop);
    }
    else
    {
        d->m_pBrowser->addProperty(prop);
    }
    d->m_pIntervalManager->blockSignals(false);
    prop->setStatusTip(param.getInfo());
    prop->setToolTip(param.getInfo());

    return ito::retOk;
}

//-----------------------------------------------------------------------
ito::RetVal ParamEditorWidget::addParamRect(const ito::Param &param, QtProperty *groupProperty)
{
    Q_D(ParamEditorWidget);

    d->m_pRectManager->blockSignals(true);
    QtProperty *prop = d->m_pRectManager->addProperty(param.getName());
    d->m_properties[param.getName()] = prop;
    prop->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));
    d->m_pRectManager->setParam(prop, param);
    if (groupProperty)
    {
        groupProperty->addSubProperty(prop);
    }
    else
    {
        d->m_pBrowser->addProperty(prop);
    }
    d->m_pRectManager->blockSignals(false);
    prop->setStatusTip(param.getInfo());
    prop->setToolTip(param.getInfo());

    return ito::retOk;
}

//-----------------------------------------------------------------------
ito::RetVal ParamEditorWidget::addParamOthers(const ito::Param &param, QtProperty *groupProperty)
{
    Q_D(ParamEditorWidget);

    d->m_pOtherManager->blockSignals(true);
    QtProperty *prop = d->m_pOtherManager->addProperty(param.getName());
    d->m_properties[param.getName()] = prop;
    prop->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));
    d->m_pOtherManager->setParam(prop, param);
    if (groupProperty)
    {
        groupProperty->addSubProperty(prop);
    }
    else
    {
        d->m_pBrowser->addProperty(prop);
    }
    d->m_pOtherManager->blockSignals(false);
    prop->setStatusTip(param.getInfo());
    prop->setToolTip(param.getInfo());

    return ito::retOk;
}

//-----------------------------------------------------------------------
void ParamEditorWidget::valueChanged(QtProperty* prop, char value)
{
    Q_D(ParamEditorWidget);
    if (!d_ptr->m_isChanging)
    {
        d->enqueue(QSharedPointer<ito::ParamBase>(new ito::ParamBase(prop->propertyName().toLatin1().data(), ito::ParamBase::Char, value)));
        if (d->m_timerID == -1)
        {
            d->m_timerID = startTimer(0);
        }
    }
}

//-----------------------------------------------------------------------
void ParamEditorWidget::valueChanged(QtProperty* prop, double value)
{
    Q_D(ParamEditorWidget);
    if (!d_ptr->m_isChanging)
    {
        d->enqueue(QSharedPointer<ito::ParamBase>(new ito::ParamBase(prop->propertyName().toLatin1().data(), ito::ParamBase::Double, value)));
        if (d->m_timerID == -1)
        {
            d->m_timerID = startTimer(0);
        }
    }
}

//-----------------------------------------------------------------------
void ParamEditorWidget::valueChanged(QtProperty* prop, int value)
{
    Q_D(ParamEditorWidget);
    if (!d_ptr->m_isChanging)
    {
        d->enqueue(QSharedPointer<ito::ParamBase>(new ito::ParamBase(prop->propertyName().toLatin1().data(), ito::ParamBase::Int, value)));
        if (d->m_timerID == -1)
        {
            d->m_timerID = startTimer(0);
        }
    }
}

//-----------------------------------------------------------------------
void ParamEditorWidget::valueChanged(QtProperty* prop, const QByteArray &value)
{
    Q_D(ParamEditorWidget);
    if (!d_ptr->m_isChanging)
    {
        d->enqueue(QSharedPointer<ito::ParamBase>(new ito::ParamBase(prop->propertyName().toLatin1().data(), ito::ParamBase::String, value.data())));
        if (d->m_timerID == -1)
        {
            d->m_timerID = startTimer(0);
        }
    }
}

//-----------------------------------------------------------------------
void ParamEditorWidget::valueChanged(QtProperty* prop, int min, int max)
{
    Q_D(ParamEditorWidget);
    if (!d_ptr->m_isChanging)
    {
        int vals[] = { min, max };
        d->enqueue(QSharedPointer<ito::ParamBase>(new ito::ParamBase(prop->propertyName().toLatin1().data(), ito::ParamBase::IntArray, 2, vals)));
        if (d->m_timerID == -1)
        {
            d->m_timerID = startTimer(0);
        }
    }
}

//-----------------------------------------------------------------------
void ParamEditorWidget::valueChanged(QtProperty* prop, int left, int top, int width, int height)
{
    Q_D(ParamEditorWidget);
    if (!d_ptr->m_isChanging)
    {
        int vals[] = { left, top, width, height };
        d->enqueue(QSharedPointer<ito::ParamBase>(new ito::ParamBase(prop->propertyName().toLatin1().data(), ito::ParamBase::IntArray, 4, vals)));
        if (d->m_timerID == -1)
        {
            d->m_timerID = startTimer(0);
        }
    }
}

//-----------------------------------------------------------------------
void ParamEditorWidget::timerEvent(QTimerEvent *event)
{
    //queueing changed parameters and processing them here is a 'hack',
    //since a direct call of setPluginParameter in the valueChanged slots
    //sometimes crashed the application (under Qt5.3, debug)
    Q_D(ParamEditorWidget);

    killTimer(d->m_timerID);
    d->m_timerID = -1;

    setPluginParameters(d->m_changedParameters);
    d->m_changedParameters.clear();
}


//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamEditorWidget::setPluginParameter(QSharedPointer<ito::ParamBase> param, MessageLevel msgLevel /*= msgLevelWarningAndError*/) const
{
    Q_D(const ParamEditorWidget);
    ito::RetVal retval;

    if (d->m_plugin)
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        if (QMetaObject::invokeMethod(d->m_plugin, "setParam", Q_ARG(QSharedPointer<ito::ParamBase>, param), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
        {
            retval += observeInvocation(locker.getSemaphore(),msgLevelNo);

            if (retval.containsWarningOrError())
            {
                QMetaObject::invokeMethod(d->m_plugin, "sendParameterRequest");
            }
        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, tr("slot 'setParam' could not be invoked since it does not exist.").toLatin1().data());
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("pointer to plugin is invalid.").toLatin1().data());
    }
    
    if (retval.containsError() && (msgLevel & msgLevelErrorOnly))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Error while setting parameter"));
        if (retval.hasErrorMessage())
        {
            msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
        }
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.exec();
    }
    else if (retval.containsWarning() && (msgLevel & msgLevelWarningOnly))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Warning while setting parameter"));
        if (retval.hasErrorMessage())
        {
            msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
        }
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.exec();
    }
    
    return retval;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamEditorWidget::setPluginParameters(const QVector<QSharedPointer<ito::ParamBase> > params, MessageLevel msgLevel /*= msgLevelWarningAndError*/) const
{
    Q_D(const ParamEditorWidget);
    ito::RetVal retval;

    if (d->m_plugin)
    {
        ItomSharedSemaphoreLocker locker(new ItomSharedSemaphore());
        if (QMetaObject::invokeMethod(d->m_plugin, "setParamVector", Q_ARG(const QVector<QSharedPointer<ito::ParamBase> >, params), Q_ARG(ItomSharedSemaphore*, locker.getSemaphore())))
        {
            retval += observeInvocation(locker.getSemaphore(),msgLevelNo);

            if (retval.containsWarningOrError())
            {
                QMetaObject::invokeMethod(d->m_plugin, "sendParameterRequest");
            }
        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, tr("slot 'setParamVector' could not be invoked since it does not exist.").toLatin1().data());
        }
    }
    else
    {
        retval += ito::RetVal(ito::retError, 0, tr("pointer to plugin is invalid.").toLatin1().data());
    }
    
    if (retval.containsError() && (msgLevel & msgLevelErrorOnly))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Error while setting parameter"));
        if (retval.hasErrorMessage())
        {
            msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
        }
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.exec();
    }
    else if (retval.containsWarning() && (msgLevel & msgLevelWarningOnly))
    {
        QMessageBox msgBox;
        msgBox.setText(tr("Warning while setting parameter"));
        if (retval.hasErrorMessage())
        {
            msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
        }
        msgBox.setIcon(QMessageBox::Warning);
        msgBox.exec();
    }
    
    return retval;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
ito::RetVal ParamEditorWidget::observeInvocation(ItomSharedSemaphore *waitCond, MessageLevel msgLevel) const
{
    Q_D(const ParamEditorWidget);
    ito::RetVal retval;
    
    if (d->m_plugin)
    {
        bool timeout = false;

        while(!timeout && waitCond->wait(PLUGINWAIT) == false)
        {
            if (d->m_plugin->isAlive() == false)
            {
                retval += ito::RetVal(ito::retError, 0, tr("Timeout while waiting for answer from plugin instance.").toLatin1().data());
                timeout = true;
            }
        }
        
        if (!timeout)
        {
            retval += waitCond->returnValue;
        }
        
        if (retval.containsError() && (msgLevel & msgLevelErrorOnly))
        {
            QMessageBox msgBox;
            msgBox.setText(tr("Error while execution"));
            if (retval.hasErrorMessage())
            {
                msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
            }
            msgBox.setIcon(QMessageBox::Critical);
            msgBox.exec();
        }
        else if (retval.containsWarning() && (msgLevel & msgLevelWarningOnly))
        {
            QMessageBox msgBox;
            msgBox.setText(tr("Warning while execution"));
            if (retval.hasErrorMessage())
            {
                msgBox.setInformativeText(QLatin1String(retval.errorMessage()));
            }
            msgBox.setIcon(QMessageBox::Warning);
            msgBox.exec();
        }
    }
    
    return retval;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------
void ParamEditorWidget::parametersChanged(QMap<QString, ito::Param> parameters)
{
    qDebug() << "drin";
    Q_D(ParamEditorWidget);
    if (!d_ptr->m_isChanging)
    {
        d_ptr->m_isChanging = true;
        QMap<QString, ito::Param>::ConstIterator it = parameters.constBegin();
        QtProperty *prop;
        ito::AbstractParamPropertyManager *manager;
        while (it != parameters.constEnd())
        {
            if (d->m_properties.contains(it->getName()))
            {
                prop = d->m_properties[it->getName()];
                manager = qobject_cast<ito::AbstractParamPropertyManager*>(prop->propertyManager());
                manager->setParam(prop, *it);
            }

            ++it;
        }
        d_ptr->m_isChanging = false;
    }
    qDebug() << "draußen";
}