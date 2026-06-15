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

#include "paramStringWidget.h"

#include <qcombobox.h>
#include <qlineedit.h>
#include <qlayout.h>
#include <qregularexpression.h>

namespace ito
{

class ParamStringWidgetPrivate
{
public:
    ParamStringWidget *q_ptr;
    ParamStringWidgetPrivate() {}

    QComboBox *m_pComboBox;
    QLineEdit *m_pLineEdit;
    ito::Param m_param;

    //--------------------------------------------------
    void slotValueChanged(const QString &value)
    {
        //value changed signal from checkbox
        emit q_ptr->valueChanged(value.toLatin1());
    }

	//--------------------------------------------------
	void slotEditingFinished()
	{
		emit q_ptr->valueChanged(m_pLineEdit->text().toLatin1());
	}

private:

    Q_DISABLE_COPY(ParamStringWidgetPrivate);
};

//-------------------------------------------------------------------------------
/*
 */
QString regExpAnchoredPattern(const QString& expression)
{
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
    return QRegularExpression::anchoredPattern(expression);
#else
    return QString() + QLatin1String("\\A(?:") + expression + QLatin1String(")\\z");
#endif
}

//-------------------------------------------------------------------------------
QString wildcardToRegularExpression(const QString& pattern)
{
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
    // conversion should be anchored, hence, strict. Not partial.
    return QRegularExpression::wildcardToRegularExpression(pattern);
#else
    // from the Qt source code
    const qsizetype wclen = pattern.size();
    QString rx;
    rx.reserve(wclen + wclen / 16);
    qsizetype i = 0;
    const QChar* wc = pattern.data();
#ifdef Q_OS_WIN
    const QLatin1Char nativePathSeparator('\\');
    const QLatin1String starEscape("[^/\\\\]*");
    const QLatin1String questionMarkEscape("[^/\\\\]");
#else
    const QLatin1Char nativePathSeparator('/');
    const QLatin1String starEscape("[^/]*");
    const QLatin1String questionMarkEscape("[^/]");
#endif
    while (i < wclen) {
        const QChar c = wc[i++];
        switch (c.unicode()) {
        case '*':
            rx += starEscape;
            break;
        case '?':
            rx += questionMarkEscape;
            break;
        case '\\':
#ifdef Q_OS_WIN
        case '/':
            rx += QLatin1String("[/\\\\]");
            break;
#endif
        case '$':
        case '(':
        case ')':
        case '+':
        case '.':
        case '^':
        case '{':
        case '|':
        case '}':
            rx += QLatin1Char('\\');
            rx += c;
            break;
        case '[':
            rx += c;
            // Support for the [!abc] or [!a-c] syntax
            if (i < wclen) {
                if (wc[i] == QLatin1Char('!')) {
                    rx += QLatin1Char('^');
                    ++i;
                }
                if (i < wclen && wc[i] == QLatin1Char(']'))
                    rx += wc[i++];
                while (i < wclen && wc[i] != QLatin1Char(']')) {
                    // The '/' appearing in a character class invalidates the
                    // regular expression parsing. It also concerns '\\' on
                    // Windows OS types.
                    if (wc[i] == QLatin1Char('/') || wc[i] == nativePathSeparator)
                        return rx;
                    if (wc[i] == QLatin1Char('\\'))
                        rx += QLatin1Char('\\');
                    rx += wc[i++];
                }
            }
            break;
        default:
            rx += c;
            break;
        }
    }
    return regExpAnchoredPattern(rx);
#endif
}

//---------------------------------------------------------------------------
ParamStringWidget::ParamStringWidget(QWidget *parent /*= NULL*/) :
    QWidget(parent),
    d_ptr(new ParamStringWidgetPrivate())
{
    d_ptr->q_ptr = this;

    Q_D(ParamStringWidget);

    d->m_param = ito::Param("", ito::ParamBase::String, "", "");
    d->m_param.setMeta(new ito::StringMeta(ito::StringMeta::String), true);

    QHBoxLayout *layout = new QHBoxLayout();
    layout->setSpacing(0);
    layout->setContentsMargins(0, 0, 0, 0);;

    d->m_pComboBox = new QComboBox(this);
    layout->addWidget(d->m_pComboBox);
    d->m_pComboBox->setVisible(false);
    connect(d->m_pComboBox, SIGNAL(currentTextChanged(QString)), this, SLOT(slotValueChanged(QString)));

    d->m_pLineEdit = new QLineEdit(this);
    layout->addWidget(d->m_pLineEdit);
    d->m_pLineEdit->setVisible(false);
	connect(d->m_pLineEdit, SIGNAL(editingFinished()), this, SLOT(slotEditingFinished()));

    setLayout(layout);
}

//---------------------------------------------------------------------------
ParamStringWidget::~ParamStringWidget()
{
}

//---------------------------------------------------------------------------
ito::Param ParamStringWidget::param() const
{
    Q_D(const ParamStringWidget);
    return d->m_param;
}

//---------------------------------------------------------------------------
void ParamStringWidget::setParam(const ito::Param &param, bool forceValueChanged /*= false*/)
{
    Q_D(ParamStringWidget);

    if (param.getType() == ito::ParamBase::String)
    {
        const ito::StringMeta *metaOld = d->m_param.getMetaT<const ito::StringMeta>(); //always != NULL
        const ito::StringMeta *metaNew = param.getMetaT<const ito::StringMeta>();

        bool valChanged = forceValueChanged || (param != d->m_param);
        bool metaChanged = ((metaNew && (*metaOld != *metaNew)) || \
                            (!metaNew));

        if (valChanged || metaChanged)
        {
            bool lineedit = !metaNew || (metaNew->getStringType() != ito::StringMeta::String) || (metaNew->getLen() <= 0);

            d->m_pLineEdit->setVisible(lineedit);
            d->m_pComboBox->setVisible(!lineedit);

            if (lineedit)
            {
                if (metaChanged && metaNew && metaNew->getLen() > 0)
                {
                    switch (metaNew->getStringType())
                    {
                    case ito::StringMeta::String:
                        d->m_pLineEdit->setValidator(nullptr);
                        break;
                    case ito::StringMeta::Wildcard:
                        {
                        QRegularExpression regexp(wildcardToRegularExpression(QLatin1String(metaNew->getString(0))));
                        d->m_pLineEdit->setValidator(new QRegularExpressionValidator(regexp, d->m_pLineEdit));
                        break;
                        }
                    case ito::StringMeta::RegExp:
                        {
                        QRegularExpression regexp(QLatin1String(metaNew->getString(0)));
                        d->m_pLineEdit->setValidator(new QRegularExpressionValidator(regexp, d->m_pLineEdit));
                        break;
                        }
                    }
                }
				else
				{
					d->m_pLineEdit->setValidator(nullptr);
				}

                if (valChanged)
                {
                    d->m_pLineEdit->blockSignals(true);
                    d->m_pLineEdit->setText(param.getVal<const char*>());
                    d->m_pLineEdit->blockSignals(false);
                }
            }
            else
            {
                if (metaChanged)
                {
                    d->m_pComboBox->clear();

                    for (int i = 0; i < metaNew->getLen(); ++i)
                    {
                        d->m_pComboBox->addItem(metaNew->getString(i));
                    }
                }

                if (valChanged)
                {
					d->m_pComboBox->setCurrentText(param.getVal<const char*>());
                }
            }

            d->m_param = param;

            if (metaNew == NULL)
            {
                d->m_param.setMeta(new ito::StringMeta(ito::StringMeta::String), true);
            }

            if (valChanged)
            {
                emit valueChanged(param.getVal<const char*>());
            }
        }
    }
}

//---------------------------------------------------------------------------
QByteArray ParamStringWidget::value() const
{
    Q_D(const ParamStringWidget);
    return d->m_param.getVal<const char*>();
}

//---------------------------------------------------------------------------
void ParamStringWidget::setValue(const QByteArray &value)
{
    Q_D(ParamStringWidget);

    if (value != QByteArray(d->m_param.getVal<const char*>()))
    {
        if (d->m_pLineEdit->isVisible())
        {
            d->m_pLineEdit->setText(value);
        }
        else
        {
			d->m_pComboBox->setCurrentText(value);
        }

        d->m_param.setVal<const char*>(value.data());

        emit valueChanged(value);
    }
}

//---------------------------------------------------------------------------
ito::StringMeta ParamStringWidget::meta() const
{
    Q_D(const ParamStringWidget);
    return *(d->m_param.getMetaT<ito::StringMeta>());
}

//---------------------------------------------------------------------------
void ParamStringWidget::setMeta(const ito::StringMeta &meta)
{
    Q_D(ParamStringWidget);
    ito::Param p = d->m_param;
    p.setMeta(new ito::StringMeta(meta), true);
    setParam(p);
}

} //end namespace ito

#include "moc_paramStringWidget.cpp"
#include "paramStringWidget.moc"
