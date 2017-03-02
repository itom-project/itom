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


#include "itomParamFactory.h"

#include "common/paramMeta.h"
#include "common/param.h"

#include "paramIntWidget.h"

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
            AbstractParamPropertyManager *manager = q_ptr->propertyManager(property);
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

} //end namespace ito

#include "moc_itomParamFactory.cpp"
#include "itomParamFactory.moc"
