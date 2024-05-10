/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2023, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

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

#ifndef WIDGETPROPGENERALLANGUAGE_H
#define WIDGETPROPGENERALLANGUAGE_H

#include "abstractPropertyPageWidget.h"

#include <qlistwidget.h>

#include <qwidget.h>

#include "ui_widgetPropGeneralLanguage.h"

namespace ito
{

class WidgetPropGeneralLanguage : public AbstractPropertyPageWidget
{
    Q_OBJECT

public:
    WidgetPropGeneralLanguage(QWidget *parent = NULL);
    ~WidgetPropGeneralLanguage();

    void readSettings();
    void writeSettings();

protected:

private:
    QString textFromLocale(const QLocale& locale) const;
    Ui::WidgetPropGeneralLanguage ui;
    QString m_operatingSystemLocale;
    QStringList m_allLocals;

signals:

public slots:

private slots:
    void on_comboLocale_currentIndexChanged(int index);
    void on_checkOmitGroupSeparator_toggled(bool checked);
};

} //end namespace ito

#endif
