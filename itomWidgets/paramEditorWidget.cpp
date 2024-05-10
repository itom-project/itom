/* ********************************************************************
itom measurement system
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2020, Institut fuer Technische Optik (ITO),
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
#include <qplaintextedit.h>
#include <qfontmetrics.h>

#include "common/addInInterface.h"

#include "qttreepropertybrowser.h"
#include "qtgroupboxpropertybrowser.h"
#include "itomParamManager.h"
#include "qtpropertymanager.h"
#include "itomParamFactory.h"

#include "collapsibleGroupBox.h"

class ParamEditorWidgetPrivate
{
public:
    ParamEditorWidgetPrivate()
        : m_pBrowser(nullptr),
        m_pTextEdit(nullptr),
        m_pInfoBox(nullptr),
        m_pIntManager(nullptr),
        m_pCharManager(nullptr),
        m_pDoubleManager(nullptr),
        m_pIntArrayManager(nullptr),
        m_pCharArrayManager(nullptr),
        m_pDoubleArrayManager(nullptr),
        m_pStringListManager(nullptr),
        m_pStringManager(nullptr),
        m_pOtherManager(nullptr),
        m_pIntervalManager(nullptr),
        m_pRectManager(nullptr),
        m_pIntFactory(nullptr),
        m_pCharFactory(nullptr),
        m_pDoubleFactory(nullptr),
        m_pStringFactory(nullptr),
        m_pIntervalFactory(nullptr),
        m_timerID(-1),
        m_isChanging(false),
        m_readonly(false),
        m_showinfo(false),
        m_immediatelyModifyPluginParametersAfterChange(true),
		m_collapsed(false)
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

        m_pIntManager->clear();
        m_pIntManager->clear();
        m_pCharManager->clear();
        m_pDoubleManager->clear();
        m_pIntArrayManager->clear();
        m_pCharArrayManager->clear();
        m_pDoubleArrayManager->clear();
        m_pStringListManager->clear();
        m_pStringManager->clear();
        m_pOtherManager->clear();
        m_pIntervalManager->clear();
        m_pRectManager->clear();
        m_properties.clear();
    }

    //-----------------------------------------------------------------------------
    void enqueue(const QSharedPointer<ito::ParamBase> param)
    {
        const char* name = param->getName();

        //check if parameter already is part of the changed-parameter set
        for (int i = 0; i < m_changedParameters.size(); ++i)
        {
            if (memcmp(m_changedParameters[i]->getName(), name, sizeof(name)) == 0)
            {
                //yes, it is part of the existing list... replace it
                m_changedParameters[i] = param;
                return;
            }
        }

        m_changedParameters.append(param);
    }

    QPointer<ito::AddInBase> m_plugin;
    QtTreePropertyBrowser *m_pBrowser;
    QPlainTextEdit *m_pTextEdit;
    CollapsibleGroupBox *m_pInfoBox;

    ito::ParamIntPropertyManager *m_pIntManager;
    ito::ParamCharPropertyManager *m_pCharManager;
    ito::ParamDoublePropertyManager *m_pDoubleManager;
    ito::ParamIntArrayPropertyManager *m_pIntArrayManager;
    ito::ParamCharArrayPropertyManager *m_pCharArrayManager;
    ito::ParamDoubleArrayPropertyManager *m_pDoubleArrayManager;
    ito::ParamStringListPropertyManager *m_pStringListManager;
    ito::ParamStringPropertyManager *m_pStringManager;
    ito::ParamOtherPropertyManager *m_pOtherManager;
    ito::ParamIntervalPropertyManager *m_pIntervalManager;
    ito::ParamRectPropertyManager *m_pRectManager;
    QtGroupPropertyManager *m_pGroupPropertyManager;

    //factories, responsible for editing properties.
    ito::ParamIntPropertyFactory *m_pIntFactory;
    ito::ParamCharPropertyFactory *m_pCharFactory;
    ito::ParamDoublePropertyFactory *m_pDoubleFactory;
    ito::ParamStringPropertyFactory *m_pStringFactory;
    ito::ParamIntervalPropertyFactory *m_pIntervalFactory;

    QVector<QSharedPointer<ito::ParamBase> > m_changedParameters;
    QMap<QByteArray, QtProperty*> m_properties;
    QMap<QString, QtProperty*> m_groups;
    int m_timerID;
    bool m_isChanging;
    bool m_readonly;
    bool m_showinfo;
    QStringList m_filteredCategories;
    bool m_immediatelyModifyPluginParametersAfterChange;
	bool m_collapsed;
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
    connect(
        d->m_pIntManager, &ito::ParamIntPropertyManager::valueChanged,
        this, static_cast<void (ParamEditorWidget::*)(QtProperty*,int)>(&ParamEditorWidget::valueChanged)
    );
    d->m_pIntFactory = new ito::ParamIntPropertyFactory(this);

    d->m_pCharManager = new ito::ParamCharPropertyManager(this);
    connect(
        d->m_pCharManager, &ito::ParamCharPropertyManager::valueChanged,
        this, static_cast<void (ParamEditorWidget::*)(QtProperty*, char)>(&ParamEditorWidget::valueChanged)
    );
    d->m_pCharFactory = new ito::ParamCharPropertyFactory(this);

    d->m_pDoubleManager = new ito::ParamDoublePropertyManager(this);
    connect(
        d->m_pDoubleManager, &ito::ParamDoublePropertyManager::valueChanged,
        this, static_cast<void (ParamEditorWidget::*)(QtProperty*, double)>(&ParamEditorWidget::valueChanged)
    );
    d->m_pDoubleFactory = new ito::ParamDoublePropertyFactory(this);

    d->m_pStringManager = new ito::ParamStringPropertyManager(this);
    connect(
        d->m_pStringManager, &ito::ParamStringPropertyManager::valueChanged,
        this, static_cast<void (ParamEditorWidget::*)(QtProperty*, const QByteArray&)>(&ParamEditorWidget::valueChanged)
    );
    d->m_pStringFactory = new ito::ParamStringPropertyFactory(this);

    d->m_pIntervalManager = new ito::ParamIntervalPropertyManager(this);
    connect(
        d->m_pIntervalManager, &ito::ParamIntervalPropertyManager::valueChanged,
        this, static_cast<void (ParamEditorWidget::*)(QtProperty*, int, int)>(&ParamEditorWidget::valueChanged)
    );
    d->m_pIntervalFactory = new ito::ParamIntervalPropertyFactory(this);

    d->m_pRectManager = new ito::ParamRectPropertyManager(this);
    connect(
        d->m_pRectManager, &ito::ParamRectPropertyManager::valueChanged,
        this, static_cast<void (ParamEditorWidget::*)(QtProperty*, int, int, int, int)>(&ParamEditorWidget::valueChanged)
    );

    d->m_pCharArrayManager = new ito::ParamCharArrayPropertyManager(this);
    connect(
        d->m_pCharArrayManager, &ito::ParamCharArrayPropertyManager::valueChanged,
        this, static_cast<void (ParamEditorWidget::*)(QtProperty*, int, const char*)>(&ParamEditorWidget::valueChanged)
    );

    d->m_pIntArrayManager = new ito::ParamIntArrayPropertyManager(this);
    connect(
        d->m_pIntArrayManager, &ito::ParamIntArrayPropertyManager::valueChanged,
        this, static_cast<void (ParamEditorWidget::*)(QtProperty*, int, const int*)>(&ParamEditorWidget::valueChanged)
    );

    d->m_pDoubleArrayManager = new ito::ParamDoubleArrayPropertyManager(this);
    connect(
        d->m_pDoubleArrayManager, &ito::ParamDoubleArrayPropertyManager::valueChanged,
        this, static_cast<void (ParamEditorWidget::*)(QtProperty*, int, const double*)>(&ParamEditorWidget::valueChanged)
    );

    d->m_pStringListManager = new ito::ParamStringListPropertyManager(this);
    connect(
        d->m_pStringListManager, &ito::ParamStringListPropertyManager::valueChanged,
        this, static_cast<void (ParamEditorWidget::*)(QtProperty*, int, const ito::ByteArray*)>(&ParamEditorWidget::valueChanged)
    );

    d->m_pOtherManager = new ito::ParamOtherPropertyManager(this);

    //the following command set all factories for the managers
    d_ptr->m_readonly = true;
    setReadonly(false);

    d->m_pTextEdit = new QPlainTextEdit(this);
    d->m_pTextEdit->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    d->m_pTextEdit->setReadOnly(true);

    QFontMetrics m(d->m_pTextEdit->font());
    int rowHeight = m.lineSpacing();
    d->m_pTextEdit->setFixedHeight(4 * rowHeight);

    connect(d->m_pBrowser, SIGNAL(currentItemChanged(QtBrowserItem*)), this, SLOT(currentItemChanged(QtBrowserItem*)));

    QVBoxLayout *vboxLayout = new QVBoxLayout();
    vboxLayout->setContentsMargins(0, 0, 0, 0);

    vboxLayout->addWidget(d->m_pBrowser);
    d->m_pBrowser->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::MinimumExpanding);
    d->m_pBrowser->sizePolicy().setVerticalStretch(20);
    d->m_pInfoBox = new CollapsibleGroupBox();
    d->m_pInfoBox->setVisible(d_ptr->m_showinfo);
    d->m_pInfoBox->setTitle("Information");
    d->m_pInfoBox->setFlat(true);
    vboxLayout->addWidget(d->m_pInfoBox);

    QVBoxLayout *vboxLayout2 = new QVBoxLayout();
    vboxLayout2->addWidget(d->m_pTextEdit);
    vboxLayout2->setContentsMargins(0, 3, 0, 0);
    d->m_pInfoBox->setLayout(vboxLayout2);
    d->m_pInfoBox->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    d->m_pInfoBox->sizePolicy().setVerticalStretch(1);

    setLayout(vboxLayout);
}

//-----------------------------------------------------------------------
ParamEditorWidget::~ParamEditorWidget()
{
    Q_D(ParamEditorWidget);
    DELETE_AND_SET_NULL(d->m_pIntFactory);
    DELETE_AND_SET_NULL(d->m_pCharFactory);
    DELETE_AND_SET_NULL(d->m_pDoubleFactory);
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
    d->m_properties.clear();
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
        loadPlugin(plugin);
    }
}

//-----------------------------------------------------------------------
ito::RetVal ParamEditorWidget::loadPlugin(QPointer<ito::AddInBase> plugin)
{
    Q_D(ParamEditorWidget);

    if (d->m_plugin)
    {
        disconnect(d->m_plugin.data(), SIGNAL(parametersChanged(QMap<QString, ito::Param>)), this, SLOT(parametersChanged(QMap<QString, ito::Param>)));
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

		QString generalName = tr("General");

        if (d->m_filteredCategories.size() == 0 || d->m_filteredCategories.contains(generalName))
        {
            d->m_groups["General"] = d->m_pGroupPropertyManager->addProperty(generalName);
            d->m_pBrowser->addProperty(d->m_groups[generalName]);
        }

        QMap<QString, ito::Param> *params;
        QMap<QString, ito::Param>::const_iterator iter;
        plugin->getParamList(&params);

        iter = params->constBegin();
        while (iter != params->constEnd())
        {
            addParam(*iter);
            ++iter;
        }

        QList<QtBrowserItem*> topLevelItems = d->m_pBrowser->topLevelItems();
        foreach (QtBrowserItem *i, topLevelItems)
        {
            foreach (QtBrowserItem* i2, i->children())
            {
                d->m_pBrowser->setExpanded(i2, false);
            }
			d->m_pBrowser->setExpanded(i, !d->m_collapsed);
        }
    }

    return ito::retOk;
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
            d->m_pBrowser->unsetFactoryForManager(d->m_pCharArrayManager->subPropertyManager());
            d->m_pBrowser->unsetFactoryForManager(d->m_pIntArrayManager->subPropertyManager());
            d->m_pBrowser->unsetFactoryForManager(d->m_pDoubleArrayManager->subPropertyManager());
            d->m_pBrowser->unsetFactoryForManager(d->m_pStringListManager->subPropertyManager());
        }
        else
        {
            d->m_pBrowser->setFactoryForManager(d->m_pIntManager, d->m_pIntFactory);
            d->m_pBrowser->setFactoryForManager(d->m_pCharManager, d->m_pCharFactory);
            d->m_pBrowser->setFactoryForManager(d->m_pDoubleManager, d->m_pDoubleFactory);
            d->m_pBrowser->setFactoryForManager(d->m_pStringManager, d->m_pStringFactory);
            d->m_pBrowser->setFactoryForManager(d->m_pIntervalManager, d->m_pIntervalFactory);
            d->m_pBrowser->setFactoryForManager(d->m_pRectManager->subIntervalPropertyManager(), d->m_pIntervalFactory);
            d->m_pBrowser->setFactoryForManager(d->m_pCharArrayManager->subPropertyManager(), d->m_pCharFactory);
            d->m_pBrowser->setFactoryForManager(d->m_pIntArrayManager->subPropertyManager(), d->m_pIntFactory);
            d->m_pBrowser->setFactoryForManager(d->m_pDoubleArrayManager->subPropertyManager(), d->m_pDoubleFactory);
            d->m_pBrowser->setFactoryForManager(d->m_pStringListManager->subPropertyManager(), d->m_pStringFactory);
        }
    }
}

//-----------------------------------------------------------------------
bool ParamEditorWidget::popupSlider() const
{
    Q_D(const ParamEditorWidget);
    return d_ptr->m_pDoubleManager->hasPopupSlider();
}

//-----------------------------------------------------------------------
void ParamEditorWidget::setPopupSlider(bool popup)
{
    Q_D(ParamEditorWidget);

    if (popup != d_ptr->m_pDoubleManager->hasPopupSlider())
    {
        d_ptr->m_pDoubleManager->setPopupSlider(popup);
    }
}

//-----------------------------------------------------------------------
void ParamEditorWidget::setShowDescriptions(bool show)
{
    Q_D(ParamEditorWidget);
    d_ptr->m_showinfo = show;
    d_ptr->m_pInfoBox->setVisible(show);
}

//-----------------------------------------------------------------------
bool ParamEditorWidget::showDescriptions() const
{
    Q_D(const ParamEditorWidget);
    return d_ptr->m_showinfo;
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
bool ParamEditorWidget::collapsed() const
{
	Q_D(const ParamEditorWidget);
	return d_ptr->m_collapsed;
}

//-----------------------------------------------------------------------
void ParamEditorWidget::setCollapsed(bool c)
{
	Q_D(ParamEditorWidget);
	if (c != d_ptr->m_collapsed)
	{
		foreach(QtBrowserItem *item, d_ptr->m_pBrowser->topLevelItems())
		{
			d_ptr->m_pBrowser->setExpanded(item, !c);
		}

		d_ptr->m_collapsed = c;
	}
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
void ParamEditorWidget::setFilteredCategories(const QStringList &filteredCategories)
{
    Q_D(ParamEditorWidget);
    if (d_ptr->m_filteredCategories != filteredCategories)
    {
        d_ptr->m_filteredCategories = filteredCategories;
        loadPlugin(d_ptr->m_plugin);
    }
}

//-----------------------------------------------------------------------
QStringList ParamEditorWidget::filteredCategories() const
{
    Q_D(const ParamEditorWidget);
    return d_ptr->m_filteredCategories;
}

//-----------------------------------------------------------------------
void ParamEditorWidget::setImmediatelyModifyPluginParamsAfterChange(bool immediateChange)
{
    Q_D(ParamEditorWidget);
    if (d_ptr->m_immediatelyModifyPluginParametersAfterChange != immediateChange)
    {
        d_ptr->m_immediatelyModifyPluginParametersAfterChange = immediateChange;

        if (immediateChange)
        {
            if (d_ptr->m_timerID == -1 && d_ptr->m_changedParameters.size() > 0)
            {
                //there are still some queued changes -> apply them now
                d_ptr->m_timerID = startTimer(0);
            }
        }
    }
}

//-----------------------------------------------------------------------
bool ParamEditorWidget::immediatelyModifyPluginParamsAfterChange() const
{
    Q_D(const ParamEditorWidget);
    return d_ptr->m_immediatelyModifyPluginParametersAfterChange;
}

//-----------------------------------------------------------------------
int ParamEditorWidget::numberOfChangedParameters() const
{
    Q_D(const ParamEditorWidget);
    return d_ptr->m_changedParameters.size();
}

//-----------------------------------------------------------------------
QVector<QSharedPointer<ito::ParamBase> > ParamEditorWidget::getAndResetChangedParameters()
{
    Q_D(ParamEditorWidget);
    QVector<QSharedPointer<ito::ParamBase> > ret;
    ret += d_ptr->m_changedParameters;
    d_ptr->m_changedParameters.clear();
    return ret;
}

//-----------------------------------------------------------------------
void ParamEditorWidget::refresh()
{
    loadPlugin(d_ptr->m_plugin);
}

//-----------------------------------------------------------------------
ito::RetVal ParamEditorWidget::addParam(const ito::Param &param)
{
    Q_D(ParamEditorWidget);

    const ito::ParamMeta *meta = param.getMetaT<ito::ParamMeta>();
    QtProperty* groupProperty = NULL;
    ito::RetVal retval;
    ito::ParamMeta::MetaRtti metaType = meta ? meta->getType() : ito::ParamMeta::rttiUnknown;

    QString group = tr("General");
    if (meta && meta->getCategory().empty() == false)
    {
        group = meta->getCategory().data();
    }

    if (d->m_filteredCategories.size() != 0 && !d->m_filteredCategories.contains(group))
    {
        //this param is part of a group that is not contained in the current filters
        return ito::retOk;
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
    case ito::ParamBase::IntArray:
        if ((metaType == ito::ParamMeta::rttiIntervalMeta) || (metaType == ito::ParamMeta::rttiRangeMeta))
        {
            retval += addParamInterval(param, groupProperty);
        }
        else if (metaType == ito::ParamMeta::rttiRectMeta)
        {
            retval += addParamRect(param, groupProperty);
        }
        else
        {
            retval += addParamIntArray(param, groupProperty);
        }
        break;
    case ito::ParamBase::CharArray:
        retval += addParamCharArray(param, groupProperty);
        break;
    case ito::ParamBase::DoubleArray:
        retval += addParamDoubleArray(param, groupProperty);
        break;
    case ito::ParamBase::StringList:
        retval += addParamStringList(param, groupProperty);
        break;
    case (ito::ParamBase::HWRef):
    case (ito::ParamBase::DObjPtr):
    case (ito::ParamBase::PointPtr):
    case (ito::ParamBase::PolygonMeshPtr):
    case (ito::ParamBase::PointCloudPtr):
    case (ito::ParamBase::Complex):
    case (ito::ParamBase::ComplexArray):
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
ito::RetVal ParamEditorWidget::addParamCharArray(const ito::Param &param, QtProperty *groupProperty)
{
    Q_D(ParamEditorWidget);

    d->m_pCharArrayManager->blockSignals(true);
    QtProperty *prop = d->m_pCharArrayManager->addProperty(param.getName());
    d->m_properties[param.getName()] = prop;
    prop->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));
    d->m_pCharArrayManager->setParam(prop, param);
    if (groupProperty)
    {
        groupProperty->addSubProperty(prop);
    }
    else
    {
        d->m_pBrowser->addProperty(prop);
    }
    d->m_pCharArrayManager->blockSignals(false);
    prop->setStatusTip(param.getInfo());
    prop->setToolTip(param.getInfo());

    return ito::retOk;
}

//-----------------------------------------------------------------------
ito::RetVal ParamEditorWidget::addParamIntArray(const ito::Param &param, QtProperty *groupProperty)
{
    Q_D(ParamEditorWidget);

    d->m_pIntArrayManager->blockSignals(true);
    QtProperty *prop = d->m_pIntArrayManager->addProperty(param.getName());
    d->m_properties[param.getName()] = prop;
    prop->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));
    d->m_pIntArrayManager->setParam(prop, param);
    if (groupProperty)
    {
        groupProperty->addSubProperty(prop);
    }
    else
    {
        d->m_pBrowser->addProperty(prop);
    }
    d->m_pIntArrayManager->blockSignals(false);
    prop->setStatusTip(param.getInfo());
    prop->setToolTip(param.getInfo());

    return ito::retOk;
}

//-----------------------------------------------------------------------
ito::RetVal ParamEditorWidget::addParamDoubleArray(const ito::Param &param, QtProperty *groupProperty)
{
    Q_D(ParamEditorWidget);

    d->m_pDoubleArrayManager->blockSignals(true);
    QtProperty *prop = d->m_pDoubleArrayManager->addProperty(param.getName());
    d->m_properties[param.getName()] = prop;
    prop->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));
    d->m_pDoubleArrayManager->setParam(prop, param);
    if (groupProperty)
    {
        groupProperty->addSubProperty(prop);
    }
    else
    {
        d->m_pBrowser->addProperty(prop);
    }
    d->m_pDoubleArrayManager->blockSignals(false);
    prop->setStatusTip(param.getInfo());
    prop->setToolTip(param.getInfo());

    return ito::retOk;
}

//-----------------------------------------------------------------------
ito::RetVal ParamEditorWidget::addParamStringList(const ito::Param &param, QtProperty *groupProperty)
{
    Q_D(ParamEditorWidget);

    d->m_pStringListManager->blockSignals(true);
    QtProperty *prop = d->m_pStringListManager->addProperty(param.getName());
    d->m_properties[param.getName()] = prop;
    prop->setEnabled(!(param.getFlags() & ito::ParamBase::Readonly));
    d->m_pStringListManager->setParam(prop, param);
    if (groupProperty)
    {
        groupProperty->addSubProperty(prop);
    }
    else
    {
        d->m_pBrowser->addProperty(prop);
    }
    d->m_pStringListManager->blockSignals(false);
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
        if (d->m_immediatelyModifyPluginParametersAfterChange && d->m_timerID == -1)
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
        if (d->m_immediatelyModifyPluginParametersAfterChange && d->m_timerID == -1)
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
        if (d->m_immediatelyModifyPluginParametersAfterChange && d->m_timerID == -1)
        {
            d->m_timerID = startTimer(0);
        }
    }
}

//-----------------------------------------------------------------------
void ParamEditorWidget::valueChanged(QtProperty* prop, int num, const char* values)
{
    Q_D(ParamEditorWidget);
    if (!d_ptr->m_isChanging)
    {
        d->enqueue(QSharedPointer<ito::ParamBase>(new ito::ParamBase(prop->propertyName().toLatin1().data(), ito::ParamBase::CharArray, num, values)));
        if (d->m_immediatelyModifyPluginParametersAfterChange && d->m_timerID == -1)
        {
            d->m_timerID = startTimer(0);
        }
    }
}

//-----------------------------------------------------------------------
void ParamEditorWidget::valueChanged(QtProperty* prop, int num, const ito::int32* values)
{
    Q_D(ParamEditorWidget);
    if (!d_ptr->m_isChanging)
    {
        d->enqueue(QSharedPointer<ito::ParamBase>(new ito::ParamBase(prop->propertyName().toLatin1().data(), ito::ParamBase::IntArray, num, values)));
        if (d->m_immediatelyModifyPluginParametersAfterChange && d->m_timerID == -1)
        {
            d->m_timerID = startTimer(0);
        }
    }
}

//-----------------------------------------------------------------------
void ParamEditorWidget::valueChanged(QtProperty* prop, int num, const ito::float64* values)
{
    Q_D(ParamEditorWidget);
    if (!d_ptr->m_isChanging)
    {
        d->enqueue(QSharedPointer<ito::ParamBase>(new ito::ParamBase(prop->propertyName().toLatin1().data(), ito::ParamBase::DoubleArray, num, values)));
        if (d->m_immediatelyModifyPluginParametersAfterChange && d->m_timerID == -1)
        {
            d->m_timerID = startTimer(0);
        }
    }
}

//-----------------------------------------------------------------------
void ParamEditorWidget::valueChanged(QtProperty* prop, int num, const ito::ByteArray* values)
{
    Q_D(ParamEditorWidget);
    if (!d_ptr->m_isChanging)
    {
        d->enqueue(QSharedPointer<ito::ParamBase>(new ito::ParamBase(prop->propertyName().toLatin1().data(), ito::ParamBase::StringList, num, values)));
        if (d->m_immediatelyModifyPluginParametersAfterChange && d->m_timerID == -1)
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
        if (d->m_immediatelyModifyPluginParametersAfterChange && d->m_timerID == -1)
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
        if (d->m_immediatelyModifyPluginParametersAfterChange && d->m_timerID == -1)
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
        if (d->m_immediatelyModifyPluginParametersAfterChange && d->m_timerID == -1)
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

    if (d->m_immediatelyModifyPluginParametersAfterChange)
    {
        setPluginParameters(d->m_changedParameters);
        d->m_changedParameters.clear();
    }
}

//-----------------------------------------------------------------------
ito::RetVal ParamEditorWidget::applyChangedParameters()
{
    Q_D(ParamEditorWidget);
    ito::RetVal retValue = setPluginParameters(d->m_changedParameters);
    d->m_changedParameters.clear();
    return retValue;
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
            QApplication::setOverrideCursor(Qt::WaitCursor);

            retval += observeInvocation(locker.getSemaphore(),msgLevelNo);

            if (retval.containsWarningOrError())
            {
                QMetaObject::invokeMethod(d->m_plugin, "sendParameterRequest");
            }

            QApplication::restoreOverrideCursor();
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
            QApplication::setOverrideCursor(Qt::WaitCursor);

            retval += observeInvocation(locker.getSemaphore(),msgLevelNo);

            if (retval.containsWarningOrError())
            {
                QMetaObject::invokeMethod(d->m_plugin, "sendParameterRequest");
            }

            QApplication::restoreOverrideCursor();
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
}


//-------------------------------------------------------------------------------------------------------------------------------------------------
void ParamEditorWidget::currentItemChanged(QtBrowserItem *item)
{
    Q_D(ParamEditorWidget);
    if (item)
    {
        d_ptr->m_pTextEdit->setPlainText(item->property()->statusTip());
    }
    else
    {
        d_ptr->m_pTextEdit->setPlainText("");
    }

}
