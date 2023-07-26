/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    University of Stuttgart, Germany

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

#define ITOM_IMPORT_API
#define ITOM_IMPORT_PLOTAPI
#include "pythonLogWidget.h"

#include <qdatetime.h>
#include <qplaintextedit.h>
#include <qlayout.h>
#include <qscrollbar.h>
#include <qmenu.h>

#include "common/sharedStructures.h"
#include "common/apiFunctionsGraphInc.h"


//-----------------------------------------------------------------------------
// PythonLogWidgetPrivate

//-----------------------------------------------------------------------------
class PythonLogWidgetPrivate
{
    Q_DECLARE_PUBLIC(PythonLogWidget);
protected:
    PythonLogWidget* const q_ptr;
public:
    PythonLogWidgetPrivate(PythonLogWidget& object) :
        q_ptr(&object),
        textEdit(NULL),
        tempText(""),
        tempType(ito::msgStreamOut),
        outputStream(true),
        errorStream(true),
        sizeHint(300, 120),
        autoScroll(true)
    {
    }

    QPlainTextEdit *textEdit;
    QString tempText;
    ito::tStreamMessageType tempType;
    bool outputStream;
    bool errorStream;
    QSize sizeHint;
    bool autoScroll;
};


//---------------------------------------------------------------------------------------------------------
PythonLogWidget::PythonLogWidget(QWidget* parent /*= NULL*/) :
    AbstractApiWidget(parent),
    d_ptr(new PythonLogWidgetPrivate(*this))
{
    Q_D(PythonLogWidget);
    d->textEdit = new QPlainTextEdit();
    d->textEdit->setObjectName("textEdit");
    d->textEdit->setReadOnly(true);
    d->textEdit->setWordWrapMode(QTextOption::NoWrap);
    d->textEdit->setUndoRedoEnabled(false);
    d->textEdit->setMaximumBlockCount(2000);

    QFont f("unexistent");
    f.setStyleHint(QFont::Monospace);
    d->textEdit->setFont(f);

    //this is for the QtDesigner only (to have something like a preview)
    d->textEdit->appendHtml(tr("This widget automatically display the selected type of python output or error messages.") +
        "<br><font color=\"red\">" + tr("Errors are displayed in red.") + "</font>");

    QHBoxLayout *layout = new QHBoxLayout();
    layout->addWidget(d->textEdit);
    layout->setContentsMargins(0,0,0,0);
    setLayout(layout);

    d->textEdit->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(d->textEdit, SIGNAL(customContextMenuRequested(const QPoint&)), SLOT(showContextMenu(const QPoint &)));


}

//---------------------------------------------------------------------------------------------------------
PythonLogWidget::~PythonLogWidget()
{
}

//---------------------------------------------------------------------------------------------------------
void PythonLogWidget::setMaxMessages(const int newMaxMessages)
{
    Q_D(PythonLogWidget);
    d->textEdit->setMaximumBlockCount(newMaxMessages);
}

//---------------------------------------------------------------------------------------------------------
void PythonLogWidget::setVerticalSizeHint(int value)
{
    Q_D(PythonLogWidget);
    d->sizeHint.setHeight(value);
    updateGeometry();
}

//---------------------------------------------------------------------------------------------------------
void PythonLogWidget::setAutoScroll(bool autoScroll)
{
    Q_D(PythonLogWidget);
    if (d->autoScroll != autoScroll)
    {
        d->autoScroll = autoScroll;
        if (d->autoScroll)
        {
            d->textEdit->verticalScrollBar()->setValue(d->textEdit->verticalScrollBar()->maximum());
        }
    }
}

//---------------------------------------------------------------------------------------------------------
int PythonLogWidget::getMaxMessages() const
{
    Q_D(const PythonLogWidget);
    return d->textEdit->maximumBlockCount();
}

//---------------------------------------------------------------------------------------------------------
bool PythonLogWidget::getOutputStream() const
{
    Q_D(const PythonLogWidget);
    return d->outputStream;
}

//---------------------------------------------------------------------------------------------------------
bool PythonLogWidget::getErrorStream() const
{
    Q_D(const PythonLogWidget);
    return d->errorStream;
}

//---------------------------------------------------------------------------------------------------------
int PythonLogWidget::getVerticalSizeHint() const
{
    return sizeHint().height();
}

//---------------------------------------------------------------------------------------------------------
bool PythonLogWidget::getAutoScroll() const
{
    Q_D(const PythonLogWidget);
    return d->autoScroll;
}

//---------------------------------------------------------------------------------------------------------
QSize PythonLogWidget::sizeHint() const
{
    Q_D(const PythonLogWidget);
    return d->sizeHint;
}

//---------------------------------------------------------------------------------------------------------
void PythonLogWidget::clear()
{
    Q_D(PythonLogWidget);
    d->textEdit->clear();
}

//---------------------------------------------------------------------------------------------------------
ito::RetVal PythonLogWidget::setOutputStream(bool enabled)
{
    Q_D(PythonLogWidget);
    ito::RetVal retval;

    if (enabled != d->outputStream)
    {
        if (ito::ITOM_API_FUNCS_GRAPH)
        {
            if (enabled)
            {
                retval += apiConnectToOutputAndErrorStream(this, SLOT(messageReceived(QString,ito::tStreamMessageType)), ito::msgStreamOut);
            }
            else
            {
                retval += apiDisconnectFromOutputAndErrorStream(this, SLOT(messageReceived(QString,ito::tStreamMessageType)), ito::msgStreamOut);
            }

        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, tr("itom plot api not available").toLatin1().data());
        }

        d->outputStream = enabled;
    }

    return retval;
}

//---------------------------------------------------------------------------------------------------------
ito::RetVal PythonLogWidget::setErrorStream(bool enabled)
{
    Q_D(PythonLogWidget);
    ito::RetVal retval;

    if (enabled != d->errorStream)
    {
        if (ito::ITOM_API_FUNCS_GRAPH)
        {
            if (enabled)
            {
                retval += apiConnectToOutputAndErrorStream(this, SLOT(messageReceived(QString,ito::tStreamMessageType)), ito::msgStreamErr);
            }
            else
            {
                retval += apiDisconnectFromOutputAndErrorStream(this, SLOT(messageReceived(QString,ito::tStreamMessageType)), ito::msgStreamErr);
            }

        }
        else
        {
            retval += ito::RetVal(ito::retError, 0, tr("itom plot api not available").toLatin1().data());
        }

        d->errorStream = enabled;
    }

    return retval;
}

//---------------------------------------------------------------------------------------------------------
ito::RetVal PythonLogWidget::init()
{
    if (ito::ITOM_API_FUNCS_GRAPH)
    {
        Q_D(const PythonLogWidget);

        ito::RetVal retval;

        if (d->errorStream)
            retval += apiConnectToOutputAndErrorStream(this, SLOT(messageReceived(QString,ito::tStreamMessageType)), ito::msgStreamErr);
        if (d->outputStream)
            retval += apiConnectToOutputAndErrorStream(this, SLOT(messageReceived(QString,ito::tStreamMessageType)), ito::msgStreamOut);

        d->textEdit->clear();

        return retval;
    }
    else
    {
        return ito::RetVal(ito::retError, 0, tr("itom plot api not available").toLatin1().data());
    }
}

//---------------------------------------------------------------------------------------------------------
void PythonLogWidget::messageReceived(QString message, ito::tStreamMessageType messageType)
{
    Q_D(PythonLogWidget);

    if (d->tempText.size() > 0 && d->tempType != messageType)
    {
        QStringList sl = d->tempText.split("\n");

        if (d->tempType == ito::msgStreamOut)
        {
            foreach (const QString &t, sl)
            {
                d->textEdit->appendHtml(t.toHtmlEscaped().replace(" ", "&nbsp;"));
            }
        }
        else
        {

            foreach (const QString &t, sl)
            {
                d->textEdit->appendHtml("<font color=\"red\">" + t.toHtmlEscaped().replace(" ", "&nbsp;") + "</font>");
            }
        }

        if (d->autoScroll)
        {
            d->textEdit->verticalScrollBar()->setValue(d->textEdit->verticalScrollBar()->maximum());
        }

        d->tempText = "";
    }

    int lastIndex = message.lastIndexOf("\n", Qt::CaseInsensitive);

    if (lastIndex >= 0)
    {
        d->tempText += message.left(lastIndex);

        QStringList sl = d->tempText.split("\n");

        if (d->tempType == ito::msgStreamOut)
        {
            foreach (const QString &t, sl)
            {
                d->textEdit->appendHtml(t.toHtmlEscaped().replace(" ", "&nbsp;"));
            }
        }
        else
        {

            foreach (const QString &t, sl)
            {
                d->textEdit->appendHtml("<font color=\"red\">" + t.toHtmlEscaped().replace(" ", "&nbsp;") + "</font>");
            }
        }

        if (d->autoScroll)
        {
            d->textEdit->verticalScrollBar()->setValue(d->textEdit->verticalScrollBar()->maximum());
        }

        d->tempText = message.mid(lastIndex+1);
        d->tempType = messageType;
    }
    else
    {
        d->tempText += message;
        d->tempType = messageType;
    }
}


//---------------------------------------------------------------------------------------------------------
void PythonLogWidget::showContextMenu(const QPoint &pt)
{
    Q_D(PythonLogWidget);

    QMenu *menu = d->textEdit->createStandardContextMenu();
    menu->addSeparator();

    QAction *actionClear = new QAction(tr("Clear"), menu);
    connect(actionClear, SIGNAL(triggered()), this, SLOT(clear()));
    menu->addAction(actionClear);

    QAction *actionAutoScroll = new QAction(tr("Auto Scroll"), menu);
    actionAutoScroll->setCheckable(true);
    actionAutoScroll->setChecked(d->autoScroll);
    connect(actionAutoScroll, SIGNAL(triggered(bool)), this, SLOT(setAutoScroll(bool)));

    menu->addAction(actionAutoScroll);

    menu->exec(d->textEdit->mapToGlobal(pt));

    DELETE_AND_SET_NULL(menu);


}
