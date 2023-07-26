/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
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


#include "itomParamFactory.h"

#include "common/paramMeta.h"
#include "common/param.h"

#include "paramIntWidget.h"
#include "paramDoubleWidget.h"
#include "paramCharWidget.h"
#include "paramStringWidget.h"
#include "rangeWidget.h"

#if defined(Q_CC_MSVC)
#    pragma warning(disable: 4786) /* MS VS 6: truncating debug info after 255 characters */
#endif

namespace ito
{

template <class Editor>
class ItomEditorFactoryPrivate
{
public:

    typedef QList<Editor *> EditorList;
    typedef QMap<QtProperty *, EditorList> PropertyToEditorListMap;
    typedef QMap<Editor *, QtProperty *> EditorToPropertyMap;

    Editor *createEditor(QtProperty *property, QWidget *parent);
    void initializeEditor(QtProperty *property, Editor *e);
    void slotEditorDestroyed(QObject *object);

    PropertyToEditorListMap  m_createdEditors;
    EditorToPropertyMap m_editorToProperty;
};

template <class Editor>
Editor *ItomEditorFactoryPrivate<Editor>::createEditor(QtProperty *property, QWidget *parent)
{
    Editor *editor = new Editor(parent);
    initializeEditor(property, editor);
    return editor;
}

template <class Editor>
void ItomEditorFactoryPrivate<Editor>::initializeEditor(QtProperty *property, Editor *editor)
{
    typename PropertyToEditorListMap::iterator it = m_createdEditors.find(property);
    if (it == m_createdEditors.end())
        it = m_createdEditors.insert(property, EditorList());
    it.value().append(editor);
    m_editorToProperty.insert(editor, property);
}

template <class Editor>
void ItomEditorFactoryPrivate<Editor>::slotEditorDestroyed(QObject *object)
{
    const typename EditorToPropertyMap::iterator ecend = m_editorToProperty.end();
    for (typename EditorToPropertyMap::iterator itEditor = m_editorToProperty.begin(); itEditor !=  ecend; ++itEditor) {
        if (itEditor.key() == object) {
            Editor *editor = itEditor.key();
            QtProperty *property = itEditor.value();
            const typename PropertyToEditorListMap::iterator pit = m_createdEditors.find(property);
            if (pit != m_createdEditors.end()) {
                pit.value().removeAll(editor);
                if (pit.value().empty())
                    m_createdEditors.erase(pit);
            }
            m_editorToProperty.erase(itEditor);
            return;
        }
    }
}

// ------------ ParamIntPropertyFactory
class ParamIntPropertyFactoryPrivate : ItomEditorFactoryPrivate<ito::ParamIntWidget>
{
    ParamIntPropertyFactory *q_ptr;
    Q_DECLARE_PUBLIC(ParamIntPropertyFactory)
public:
    void slotPropertyChanged(QtProperty *property, int value);
    void slotMetaChanged(QtProperty *property, const ito::IntMeta &meta);
    void slotSetValue(int value);
};

void ParamIntPropertyFactoryPrivate::slotPropertyChanged(QtProperty *property, int value)
{
    if (!m_createdEditors.contains(property))
        return;
    QListIterator<ito::ParamIntWidget *> itEditor(m_createdEditors[property]);
    while (itEditor.hasNext()) {
        ito::ParamIntWidget *editor = itEditor.next();
        if (editor->value() != value) {
            editor->blockSignals(true);
            editor->setValue(value);
            editor->blockSignals(false);
        }
    }
}

void ParamIntPropertyFactoryPrivate::slotMetaChanged(QtProperty *property, const ito::IntMeta &meta)
{
    if (!m_createdEditors.contains(property))
        return;

    ParamIntPropertyManager *manager = q_ptr->propertyManager(property);
    if (!manager)
        return;

    QListIterator<ito::ParamIntWidget *> itEditor(m_createdEditors[property]);
    while (itEditor.hasNext()) {
        ito::ParamIntWidget *editor = itEditor.next();
        editor->blockSignals(true);
        editor->setMeta(meta);
        editor->setValue(manager->paramBase(property).getVal<int>());
        editor->blockSignals(false);
    }
}

void ParamIntPropertyFactoryPrivate::slotSetValue(int value)
{
    QObject *object = q_ptr->sender();
    const QMap<ito::ParamIntWidget *, QtProperty *>::ConstIterator ecend = m_editorToProperty.constEnd();
    for (QMap<ito::ParamIntWidget *, QtProperty *>::ConstIterator itEditor = m_editorToProperty.constBegin(); itEditor != ecend; ++itEditor ) {
        if (itEditor.key() == object) {
            QtProperty *property = itEditor.value();
            ParamIntPropertyManager *manager = q_ptr->propertyManager(property);
            if (!manager)
                return;
            manager->setValue(property, value);
            return;
        }
    }
}

/*!
    \class QtSpinBoxFactory

    \brief The QtSpinBoxFactory class provides QSpinBox widgets for
    properties created by QtIntPropertyManager objects.

    \sa QtAbstractEditorFactory, QtIntPropertyManager
*/

/*!
    Creates a factory with the given \a parent.
*/
ParamIntPropertyFactory::ParamIntPropertyFactory(QObject *parent)
    : QtAbstractEditorFactory<ParamIntPropertyManager>(parent)
{
    d_ptr = new ParamIntPropertyFactoryPrivate();
    d_ptr->q_ptr = this;

}

/*!
    Destroys this factory, and all the widgets it has created.
*/
ParamIntPropertyFactory::~ParamIntPropertyFactory()
{
    qDeleteAll(d_ptr->m_editorToProperty.keys());
    delete d_ptr;
}

/*!
    \internal

    Reimplemented from the QtAbstractEditorFactory class.
*/
void ParamIntPropertyFactory::connectPropertyManager(ParamIntPropertyManager *manager)
{
    connect(manager, SIGNAL(valueChanged(QtProperty *, int)),
                this, SLOT(slotPropertyChanged(QtProperty *, int)));
    connect(manager, SIGNAL(metaChanged(QtProperty *, ito::IntMeta)),
                this, SLOT(slotMetaChanged(QtProperty *, ito::IntMeta)));
}

/*!
    \internal

    Reimplemented from the QtAbstractEditorFactory class.
*/
QWidget *ParamIntPropertyFactory::createEditor(ParamIntPropertyManager *manager, QtProperty *property,
        QWidget *parent)
{
    ito::ParamIntWidget *editor = d_ptr->createEditor(property, parent);
    const ito::Param &param = manager->param(property);
    editor->setParam(param, true);
    editor->setKeyboardTracking(false);

    connect(editor, SIGNAL(valueChanged(int)), this, SLOT(slotSetValue(int)));
    connect(editor, SIGNAL(destroyed(QObject *)),
                this, SLOT(slotEditorDestroyed(QObject *)));
    return editor;
}

/*!
    \internal

    Reimplemented from the QtAbstractEditorFactory class.
*/
void ParamIntPropertyFactory::disconnectPropertyManager(ParamIntPropertyManager *manager)
{
    disconnect(manager, SIGNAL(valueChanged(QtProperty *, int)),
                this, SLOT(slotPropertyChanged(QtProperty *, int)));
    disconnect(manager, SIGNAL(metaChanged(QtProperty *, ito::IntMeta)),
                this, SLOT(slotMetaChanged(QtProperty *, ito::IntMeta)));
}






// ------------ ParamIntPropertyFactory
class ParamDoublePropertyFactoryPrivate : ItomEditorFactoryPrivate<ito::ParamDoubleWidget>
{
    ParamDoublePropertyFactory *q_ptr;
    Q_DECLARE_PUBLIC(ParamDoublePropertyFactory)
public:
    void slotPropertyChanged(QtProperty *property, double value);
    void slotMetaChanged(QtProperty *property, const ito::DoubleMeta &meta);
    void slotSetValue(double value);
};

void ParamDoublePropertyFactoryPrivate::slotPropertyChanged(QtProperty *property, double value)
{
    if (!m_createdEditors.contains(property))
        return;
    QListIterator<ito::ParamDoubleWidget *> itEditor(m_createdEditors[property]);
    while (itEditor.hasNext()) {
        ito::ParamDoubleWidget *editor = itEditor.next();
        if (editor->value() != value) {
            editor->blockSignals(true);
            editor->setValue(value);
            editor->blockSignals(false);
        }
    }
}

void ParamDoublePropertyFactoryPrivate::slotMetaChanged(QtProperty *property, const ito::DoubleMeta &meta)
{
    if (!m_createdEditors.contains(property))
        return;

    ParamDoublePropertyManager *manager = q_ptr->propertyManager(property);
    if (!manager)
        return;

    QListIterator<ito::ParamDoubleWidget *> itEditor(m_createdEditors[property]);
    while (itEditor.hasNext()) {
        ito::ParamDoubleWidget *editor = itEditor.next();
        editor->blockSignals(true);
        editor->setMeta(meta);
        editor->setValue(manager->paramBase(property).getVal<double>());
        editor->blockSignals(false);
    }
}

void ParamDoublePropertyFactoryPrivate::slotSetValue(double value)
{
    QObject *object = q_ptr->sender();
    const QMap<ito::ParamDoubleWidget *, QtProperty *>::ConstIterator ecend = m_editorToProperty.constEnd();
    for (QMap<ito::ParamDoubleWidget *, QtProperty *>::ConstIterator itEditor = m_editorToProperty.constBegin(); itEditor != ecend; ++itEditor) {
        if (itEditor.key() == object) {
            QtProperty *property = itEditor.value();
            ParamDoublePropertyManager *manager = q_ptr->propertyManager(property);
            if (!manager)
                return;
            manager->setValue(property, value);
            return;
        }
    }
}

/*!
\class ParamDoublePropertyFactory

\brief The ParamDoublePropertyFactory class provides QSpinBox widgets for
properties created by QtIntPropertyManager objects.

\sa QtAbstractEditorFactory, QtIntPropertyManager
*/

/*!
Creates a factory with the given \a parent.
*/
ParamDoublePropertyFactory::ParamDoublePropertyFactory(QObject *parent)
    : QtAbstractEditorFactory<ParamDoublePropertyManager>(parent)
{
    d_ptr = new ParamDoublePropertyFactoryPrivate();
    d_ptr->q_ptr = this;

}

/*!
Destroys this factory, and all the widgets it has created.
*/
ParamDoublePropertyFactory::~ParamDoublePropertyFactory()
{
    qDeleteAll(d_ptr->m_editorToProperty.keys());
    delete d_ptr;
}

/*!
\internal

Reimplemented from the QtAbstractEditorFactory class.
*/
void ParamDoublePropertyFactory::connectPropertyManager(ParamDoublePropertyManager *manager)
{
    connect(manager, SIGNAL(valueChanged(QtProperty *, double)),
        this, SLOT(slotPropertyChanged(QtProperty *, double)));
    connect(manager, SIGNAL(metaChanged(QtProperty *, ito::DoubleMeta)),
        this, SLOT(slotMetaChanged(QtProperty *, ito::DoubleMeta)));
}

/*!
\internal

Reimplemented from the QtAbstractEditorFactory class.
*/
QWidget *ParamDoublePropertyFactory::createEditor(ParamDoublePropertyManager *manager, QtProperty *property,
    QWidget *parent)
{
    ito::ParamDoubleWidget *editor = d_ptr->createEditor(property, parent);
    const ito::Param &param = manager->param(property);
    editor->setParam(param, true);
    editor->setKeyboardTracking(false);
    editor->setPopupSlider(manager->hasPopupSlider());

    connect(editor, SIGNAL(valueChanged(double)), this, SLOT(slotSetValue(double)));
    connect(editor, SIGNAL(destroyed(QObject *)),
        this, SLOT(slotEditorDestroyed(QObject *)));
    return editor;
}

/*!
\internal

Reimplemented from the QtAbstractEditorFactory class.
*/
void ParamDoublePropertyFactory::disconnectPropertyManager(ParamDoublePropertyManager *manager)
{
    disconnect(manager, SIGNAL(valueChanged(QtProperty *, double)),
        this, SLOT(slotPropertyChanged(QtProperty *, double)));
    disconnect(manager, SIGNAL(metaChanged(QtProperty *, ito::DoubleMeta)),
        this, SLOT(slotMetaChanged(QtProperty *, ito::DoubleMeta)));
}











// ------------ ParamCharPropertyFactory
class ParamCharPropertyFactoryPrivate : ItomEditorFactoryPrivate<ito::ParamCharWidget>
{
    ParamCharPropertyFactory *q_ptr;
    Q_DECLARE_PUBLIC(ParamCharPropertyFactory)
public:
    void slotPropertyChanged(QtProperty *property, char value);
    void slotMetaChanged(QtProperty *property, const ito::CharMeta &meta);
    void slotSetValue(char value);
};

void ParamCharPropertyFactoryPrivate::slotPropertyChanged(QtProperty *property, char value)
{
    if (!m_createdEditors.contains(property))
        return;
    QListIterator<ito::ParamCharWidget *> itEditor(m_createdEditors[property]);
    while (itEditor.hasNext()) {
        ito::ParamCharWidget *editor = itEditor.next();
        if (editor->value() != value) {
            editor->blockSignals(true);
            editor->setValue(value);
            editor->blockSignals(false);
        }
    }
}

void ParamCharPropertyFactoryPrivate::slotMetaChanged(QtProperty *property, const ito::CharMeta &meta)
{
    if (!m_createdEditors.contains(property))
        return;

    ParamCharPropertyManager *manager = q_ptr->propertyManager(property);
    if (!manager)
        return;

    QListIterator<ito::ParamCharWidget *> itEditor(m_createdEditors[property]);
    while (itEditor.hasNext()) {
        ito::ParamCharWidget *editor = itEditor.next();
        editor->blockSignals(true);
        editor->setMeta(meta);
        editor->setValue(manager->paramBase(property).getVal<char>());
        editor->blockSignals(false);
    }
}

void ParamCharPropertyFactoryPrivate::slotSetValue(char value)
{
    QObject *object = q_ptr->sender();
    const QMap<ito::ParamCharWidget *, QtProperty *>::ConstIterator ecend = m_editorToProperty.constEnd();
    for (QMap<ito::ParamCharWidget *, QtProperty *>::ConstIterator itEditor = m_editorToProperty.constBegin(); itEditor != ecend; ++itEditor) {
        if (itEditor.key() == object) {
            QtProperty *property = itEditor.value();
            ParamCharPropertyManager *manager = q_ptr->propertyManager(property);
            if (!manager)
                return;
            manager->setValue(property, value);
            return;
        }
    }
}

/*!
\class QtSpinBoxFactory

\brief The QtSpinBoxFactory class provides QSpinBox widgets for
properties created by QtIntPropertyManager objects.

\sa QtAbstractEditorFactory, QtIntPropertyManager
*/

/*!
Creates a factory with the given \a parent.
*/
ParamCharPropertyFactory::ParamCharPropertyFactory(QObject *parent)
    : QtAbstractEditorFactory<ParamCharPropertyManager>(parent)
{
    d_ptr = new ParamCharPropertyFactoryPrivate();
    d_ptr->q_ptr = this;

}

/*!
Destroys this factory, and all the widgets it has created.
*/
ParamCharPropertyFactory::~ParamCharPropertyFactory()
{
    qDeleteAll(d_ptr->m_editorToProperty.keys());
    delete d_ptr;
}

/*!
\internal

Reimplemented from the QtAbstractEditorFactory class.
*/
void ParamCharPropertyFactory::connectPropertyManager(ParamCharPropertyManager *manager)
{
    connect(manager, SIGNAL(valueChanged(QtProperty *, char)),
        this, SLOT(slotPropertyChanged(QtProperty *, char)));
    connect(manager, SIGNAL(metaChanged(QtProperty *, ito::CharMeta)),
        this, SLOT(slotMetaChanged(QtProperty *, ito::CharMeta)));
}

/*!
\internal

Reimplemented from the QtAbstractEditorFactory class.
*/
QWidget *ParamCharPropertyFactory::createEditor(ParamCharPropertyManager *manager, QtProperty *property,
    QWidget *parent)
{
    ito::ParamCharWidget *editor = d_ptr->createEditor(property, parent);
    const ito::Param &param = manager->param(property);
    editor->setParam(param, true);
    editor->setKeyboardTracking(false);

    connect(editor, SIGNAL(valueChanged(char)), this, SLOT(slotSetValue(char)));
    connect(editor, SIGNAL(destroyed(QObject *)),
        this, SLOT(slotEditorDestroyed(QObject *)));
    return editor;
}

/*!
\internal

Reimplemented from the QtAbstractEditorFactory class.
*/
void ParamCharPropertyFactory::disconnectPropertyManager(ParamCharPropertyManager *manager)
{
    disconnect(manager, SIGNAL(valueChanged(QtProperty *, char)),
        this, SLOT(slotPropertyChanged(QtProperty *, char)));
    disconnect(manager, SIGNAL(metaChanged(QtProperty *, ito::CharMeta)),
        this, SLOT(slotMetaChanged(QtProperty *, ito::CharMeta)));
}









// ------------ ParamStringPropertyFactory
class ParamStringPropertyFactoryPrivate : ItomEditorFactoryPrivate<ito::ParamStringWidget>
{
    ParamStringPropertyFactory *q_ptr;
    Q_DECLARE_PUBLIC(ParamStringPropertyFactory)
public:
    void slotPropertyChanged(QtProperty *property, const QByteArray &value);
    void slotMetaChanged(QtProperty *property, const ito::StringMeta &meta);
    void slotSetValue(const QByteArray &value);
};

void ParamStringPropertyFactoryPrivate::slotPropertyChanged(QtProperty *property, const QByteArray &value)
{
    if (!m_createdEditors.contains(property))
        return;
    QListIterator<ito::ParamStringWidget *> itEditor(m_createdEditors[property]);
    while (itEditor.hasNext()) {
        ito::ParamStringWidget *editor = itEditor.next();
        if (editor->value() != value) {
            editor->blockSignals(true);
            editor->setValue(value);
            editor->blockSignals(false);
        }
    }
}

void ParamStringPropertyFactoryPrivate::slotMetaChanged(QtProperty *property, const ito::StringMeta &meta)
{
    if (!m_createdEditors.contains(property))
        return;

    ParamStringPropertyManager *manager = q_ptr->propertyManager(property);
    if (!manager)
        return;

    QListIterator<ito::ParamStringWidget *> itEditor(m_createdEditors[property]);
    while (itEditor.hasNext()) {
        ito::ParamStringWidget *editor = itEditor.next();
        editor->blockSignals(true);
        editor->setMeta(meta);
        editor->setValue(QByteArray(manager->paramBase(property).getVal<const char*>()));
        editor->blockSignals(false);
    }
}

void ParamStringPropertyFactoryPrivate::slotSetValue(const QByteArray &value)
{
    QObject *object = q_ptr->sender();
    const QMap<ito::ParamStringWidget *, QtProperty *>::ConstIterator ecend = m_editorToProperty.constEnd();
    for (QMap<ito::ParamStringWidget *, QtProperty *>::ConstIterator itEditor = m_editorToProperty.constBegin(); itEditor != ecend; ++itEditor) {
        if (itEditor.key() == object) {
            QtProperty *property = itEditor.value();
            ParamStringPropertyManager *manager = q_ptr->propertyManager(property);
            if (!manager)
                return;
            manager->setValue(property, value);
            return;
        }
    }
}

/*!
\class ParamStringPropertyFactory

\brief The ParamStringPropertyFactory class provides QLineEdit widgets for
properties created by ParamStringPropertyManager objects.

\sa QtAbstractEditorFactory, ParamStringPropertyManager
*/

/*!
Creates a factory with the given \a parent.
*/
ParamStringPropertyFactory::ParamStringPropertyFactory(QObject *parent)
    : QtAbstractEditorFactory<ParamStringPropertyManager>(parent)
{
    d_ptr = new ParamStringPropertyFactoryPrivate();
    d_ptr->q_ptr = this;

}

/*!
Destroys this factory, and all the widgets it has created.
*/
ParamStringPropertyFactory::~ParamStringPropertyFactory()
{
    qDeleteAll(d_ptr->m_editorToProperty.keys());
    delete d_ptr;
}

/*!
\internal

Reimplemented from the QtAbstractEditorFactory class.
*/
void ParamStringPropertyFactory::connectPropertyManager(ParamStringPropertyManager *manager)
{
    connect(manager, SIGNAL(valueChanged(QtProperty *, QByteArray)),
        this, SLOT(slotPropertyChanged(QtProperty *, QByteArray)));
    connect(manager, SIGNAL(metaChanged(QtProperty *, ito::StringMeta)),
        this, SLOT(slotMetaChanged(QtProperty *, ito::StringMeta)));
}

/*!
\internal

Reimplemented from the QtAbstractEditorFactory class.
*/
QWidget *ParamStringPropertyFactory::createEditor(ParamStringPropertyManager *manager, QtProperty *property,
    QWidget *parent)
{
    ito::ParamStringWidget *editor = d_ptr->createEditor(property, parent);
    const ito::Param &param = manager->param(property);
    editor->setParam(param, true);

    connect(editor, SIGNAL(valueChanged(QByteArray)), this, SLOT(slotSetValue(QByteArray)));
    connect(editor, SIGNAL(destroyed(QObject *)),
        this, SLOT(slotEditorDestroyed(QObject *)));
    return editor;
}

/*!
\internal

Reimplemented from the QtAbstractEditorFactory class.
*/
void ParamStringPropertyFactory::disconnectPropertyManager(ParamStringPropertyManager *manager)
{
    disconnect(manager, SIGNAL(valueChanged(QtProperty *, QByteArray)),
        this, SLOT(slotPropertyChanged(QtProperty *, QByteArray)));
    disconnect(manager, SIGNAL(metaChanged(QtProperty *, ito::StringMeta)),
        this, SLOT(slotMetaChanged(QtProperty *, ito::StringMeta)));
}



// ------------ ParamIntervalPropertyFactoryPrivate
class ParamIntervalPropertyFactoryPrivate : ItomEditorFactoryPrivate<RangeWidget>
{
    ParamIntervalPropertyFactory *q_ptr;
    Q_DECLARE_PUBLIC(ParamIntervalPropertyFactory)
public:
    void slotPropertyChanged(QtProperty *property, int min, int max);
    void slotMetaChanged(QtProperty *property, const ito::IntervalMeta &meta);
    void slotSetValue(int min, int max);
};

void ParamIntervalPropertyFactoryPrivate::slotPropertyChanged(QtProperty *property, int min, int max)
{
    if (!m_createdEditors.contains(property))
        return;
    QListIterator<RangeWidget *> itEditor(m_createdEditors[property]);
    while (itEditor.hasNext()) {
        RangeWidget *editor = itEditor.next();
        if (editor->minimum() != min || editor->maximum() != max) {
            editor->blockSignals(true);
            editor->setValues(min, max);
            editor->blockSignals(false);
        }
    }
}

void ParamIntervalPropertyFactoryPrivate::slotMetaChanged(QtProperty *property, const ito::IntervalMeta &meta)
{
    if (!m_createdEditors.contains(property))
        return;

    ParamIntervalPropertyManager *manager = q_ptr->propertyManager(property);
    if (!manager)
        return;

    QListIterator<RangeWidget *> itEditor(m_createdEditors[property]);
    while (itEditor.hasNext()) {
        RangeWidget *editor = itEditor.next();
        editor->blockSignals(true);
        editor->setLimitsFromIntervalMeta(meta);
        const int* vals = manager->paramBase(property).getVal<const int*>();
        editor->setValues(vals[0], vals[1]);
        editor->blockSignals(false);
    }
}

void ParamIntervalPropertyFactoryPrivate::slotSetValue(int min, int max)
{
    QObject *object = q_ptr->sender();
    const QMap<RangeWidget *, QtProperty *>::ConstIterator ecend = m_editorToProperty.constEnd();
    for (QMap<RangeWidget *, QtProperty *>::ConstIterator itEditor = m_editorToProperty.constBegin(); itEditor != ecend; ++itEditor) {
        if (itEditor.key() == object) {
            QtProperty *property = itEditor.value();
            ParamIntervalPropertyManager *manager = q_ptr->propertyManager(property);
            if (!manager)
                return;
            manager->setValue(property, min, max);
            return;
        }
    }
}


/*!
\class ParamIntervalPropertyFactory

\brief The ParamIntervalPropertyFactory class provides RangeWidget widgets for
properties created by ParamIntervalPropertyManager objects.

\sa QtAbstractEditorFactory, ParamIntervalPropertyManager
*/

/*!
Creates a factory with the given \a parent.
*/
ParamIntervalPropertyFactory::ParamIntervalPropertyFactory(QObject *parent)
    : QtAbstractEditorFactory<ParamIntervalPropertyManager>(parent)
{
    d_ptr = new ParamIntervalPropertyFactoryPrivate();
    d_ptr->q_ptr = this;

}

/*!
Destroys this factory, and all the widgets it has created.
*/
ParamIntervalPropertyFactory::~ParamIntervalPropertyFactory()
{
    qDeleteAll(d_ptr->m_editorToProperty.keys());
    delete d_ptr;
}

/*!
\internal

Reimplemented from the QtAbstractEditorFactory class.
*/
void ParamIntervalPropertyFactory::connectPropertyManager(ParamIntervalPropertyManager *manager)
{
    connect(manager, SIGNAL(valueChanged(QtProperty *, int, int)),
        this, SLOT(slotPropertyChanged(QtProperty *, int, int)));
    connect(manager, SIGNAL(metaChanged(QtProperty *, ito::IntervalMeta)),
        this, SLOT(slotMetaChanged(QtProperty *, ito::IntervalMeta)));
}

/*!
\internal

Reimplemented from the QtAbstractEditorFactory class.
*/
QWidget *ParamIntervalPropertyFactory::createEditor(ParamIntervalPropertyManager *manager, QtProperty *property,
    QWidget *parent)
{
    RangeWidget *editor = d_ptr->createEditor(property, parent);
    editor->setTracking(false);
    const ito::Param &param = manager->param(property);
    const int* vals = param.getVal<const int*>();
    editor->setLimitsFromIntervalMeta(*(param.getMetaT<ito::IntervalMeta>()));
    editor->setValues(vals[0], vals[1]);

    connect(editor, SIGNAL(valuesChanged(int, int)), this, SLOT(slotSetValue(int, int)));
    connect(editor, SIGNAL(destroyed(QObject *)),
        this, SLOT(slotEditorDestroyed(QObject *)));
    return editor;
}

/*!
\internal

Reimplemented from the QtAbstractEditorFactory class.
*/
void ParamIntervalPropertyFactory::disconnectPropertyManager(ParamIntervalPropertyManager *manager)
{
    disconnect(manager, SIGNAL(valueChanged(QtProperty *, int, int)),
        this, SLOT(slotPropertyChanged(QtProperty *, int, int)));
    disconnect(manager, SIGNAL(metaChanged(QtProperty *, ito::IntervalMeta)),
        this, SLOT(slotMetaChanged(QtProperty *, ito::IntervalMeta)));
}

} //end namespace ito

#include "moc_itomParamFactory.cpp"
#include "itomParamFactory.moc"
