/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2013, Institut für Technische Optik (ITO),
    Universität Stuttgart, Germany

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

#include "userUiDialog.h"

#include "../global.h"
#include "../../common/apiFunctionsGraphInc.h"
#include "plot/AbstractFigure.h"

#include <QtUiTools/quiloader.h>
#include <qcoreapplication.h>
#include <qfile.h>
#include <qlayout.h>
#include <qdebug.h>
#include <qmetaobject.h>
#include <iostream>
#include <qdir.h>


//----------------------------------------------------------------------------------------------------------------------------------
UserUiDialog::UserUiDialog(QString filename, tButtonBarType buttonBarType, QMap<QString,QString> &dialogButtons, RetVal &retValue, QWidget * parent, Qt::WindowFlags f) :
    QDialog(parent, f),
    m_boxLayout(NULL),
    m_dialogBtnBox(NULL),
    m_uiWidget(NULL)
{
    retValue = init(filename, buttonBarType, dialogButtons);
}

//----------------------------------------------------------------------------------------------------------------------------------
UserUiDialog::UserUiDialog(QWidget *contentWidget, tButtonBarType buttonBarType, QMap<QString,QString> &dialogButtons, RetVal &retValue, QWidget * parent, Qt::WindowFlags f) :
    QDialog(parent, f),
    m_boxLayout(NULL),
    m_dialogBtnBox(NULL),
    m_uiWidget(NULL)
{
    retValue = init(contentWidget, buttonBarType, dialogButtons);
}

//----------------------------------------------------------------------------------------------------------------------------------
UserUiDialog::~UserUiDialog()
{
    if (m_uiWidget)
    {
        m_uiWidget->deleteLater();
        m_uiWidget = NULL;
    }

    if (m_dialogBtnBox)
    {
        m_dialogBtnBox->deleteLater();
        m_dialogBtnBox = NULL;
    }

    DELETE_AND_SET_NULL(m_boxLayout);

}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UserUiDialog::init(QString filename, tButtonBarType buttonBarType, QMap<QString,QString> &dialogButtons)
{
    RetVal retValue(retOk);

    QUiLoader loader;

    QFile file(QDir::cleanPath(filename));
    QWidget *contentWidget = NULL;
    if (!file.exists())
    {
        m_uiWidget = NULL;
        retValue += RetVal::format(retError, 1006, tr("filename '%s' does not exist").toAscii().data(), filename.toAscii().data());
    }
    else
    {
		//set the working directory if QLoader to the directory where the ui-file is stored. Then icons, assigned to the user-interface may be properly loaded, since their path is always saved relatively to the ui-file,too.
        file.open(QFile::ReadOnly);
		QFileInfo fileinfo(filename);
		QDir workingDirectory = fileinfo.absoluteDir();
		loader.setWorkingDirectory(workingDirectory);
		//qDebug() << "working dir of QLoader:" << loader.workingDirectory();

        contentWidget = loader.load(&file, NULL);
        file.close();

        if (contentWidget == NULL)
        {
            retValue += RetVal(retError, 1007, tr("ui-file could not be correctly parsed.").toAscii().data());
        }

		retValue += init(contentWidget, buttonBarType, dialogButtons);
    }

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
RetVal UserUiDialog::init(QWidget *contentWidget, tButtonBarType buttonBarType, QMap<QString,QString> &dialogButtons)
{
    RetVal retValue(retOk);

    if(contentWidget->windowTitle() != "")
	{
		setWindowTitle(contentWidget->windowTitle());
	}
	else
	{
		setWindowTitle(tr("itom"));
	}

    QObject *child = NULL;
    foreach(child, contentWidget->children())
    {
        if (child->inherits("ito::AbstractFigure"))
        {
            ((ito::AbstractFigure*)child)->setApiFunctionBasePtr(ITOM_API_FUNCS);
            ((ito::AbstractFigure*)child)->setApiFunctionGraphBasePtr(ITOM_API_FUNCS_GRAPH);
        }
    }

    contentWidget->setWindowFlags(Qt::Widget);
    m_uiWidget = contentWidget;
    

    if (m_uiWidget == NULL)
    {
        retValue += RetVal(retError, 1007, tr("content-widget is empty.").toAscii().data());
    }

    if (buttonBarType & (UserUiDialog::bbTypeHorizontal | UserUiDialog::bbTypeVertical))
    {
        m_dialogBtnBox = new QDialogButtonBox(this);
        m_dialogBtnBox->setObjectName("dialogButtonBox");
        connect(m_dialogBtnBox, SIGNAL(clicked (QAbstractButton*)), this, SLOT(dialogButtonClicked(QAbstractButton*)));

        QMap<QString, QString>::const_iterator i = dialogButtons.constBegin();
        while (i != dialogButtons.constEnd())
        {
            QDialogButtonBox::ButtonRole role = getButtonRole(i.key());
            if (role == QDialogButtonBox::InvalidRole)
            {
                retValue += RetVal(retWarning, 1004, tr("dialog button role is unknown").toAscii().data());
            }
            m_dialogBtnBox->addButton(i.value(), role);
            ++i;
        }

        if (buttonBarType & UserUiDialog::bbTypeHorizontal) //horizontal
        {
            m_dialogBtnBox->setOrientation(Qt::Horizontal);
            m_boxLayout = new QVBoxLayout();
        }
        else //vertical
        {
            m_dialogBtnBox->setOrientation(Qt::Vertical);
            m_boxLayout = new QHBoxLayout();
        }

        //m_boxLayout->setContentsMargins(2,2,2,2);
        if (m_uiWidget) m_boxLayout->addWidget(m_uiWidget);
        m_boxLayout->addWidget(m_dialogBtnBox);
    }
    else //no button bar
    {
        m_boxLayout = new QVBoxLayout();
        //m_boxLayout->setContentsMargins(2,2,2,2);
        if (m_uiWidget) m_boxLayout->addWidget(m_uiWidget);
    }

    this->setLayout(m_boxLayout);

    return retValue;
}

//----------------------------------------------------------------------------------------------------------------------------------
QDialogButtonBox::ButtonRole UserUiDialog::getButtonRole(QString role)
{
    if (QString::compare(role, "AcceptRole", Qt::CaseInsensitive) == 0)
    {
        return QDialogButtonBox::AcceptRole;
    }
    else if (QString::compare(role, "RejectRole", Qt::CaseInsensitive) == 0)
    {
        return QDialogButtonBox::RejectRole;
    }
    else if (QString::compare(role, "YesRole", Qt::CaseInsensitive) == 0)
    {
        return QDialogButtonBox::YesRole;
    }
    else if (QString::compare(role, "NoRole", Qt::CaseInsensitive) == 0)
    {
        return QDialogButtonBox::NoRole;
    }
    else if (QString::compare(role, "ApplyRole", Qt::CaseInsensitive) == 0)
    {
        return QDialogButtonBox::ApplyRole;
    }
    else
    {
        return QDialogButtonBox::InvalidRole;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
void UserUiDialog::dialogButtonClicked (QAbstractButton * button)
{
    QDialogButtonBox::ButtonRole role = m_dialogBtnBox->buttonRole(button);

    switch(role)
    {
    case QDialogButtonBox::AcceptRole:
        this->accept();
        break;
    case QDialogButtonBox::RejectRole:
        this->reject();
        break;
    }
}

//----------------------------------------------------------------------------------------------------------------------------------
const QMetaObject* UserUiDialog::getMetaObjectByWidgetName(QString name) const
{
    QWidget *widget = findChild<QWidget*>(name);
    const QMetaObject *mo = NULL;
    if (widget)
    {
        mo = widget->metaObject();
    }
    return mo;
}