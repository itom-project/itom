/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
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

#ifndef DIALOGABOUTQITOM
#define DIALOGABOUTQITOM

#include <QtGui>
#include <qdialog.h>
#include <qcolor.h>

#include "ui_dialogAbout.h"

namespace ito
{

class DialogAboutQItom : public QDialog
{
    Q_OBJECT

    Q_PROPERTY(QColor textColor READ textColor WRITE setTextColor DESIGNABLE true);
    Q_PROPERTY(QColor linkColor READ linkColor WRITE setLinkColor DESIGNABLE true);

    public:
        DialogAboutQItom(const QMap<QString, QString> &versionMap, QWidget *parent = NULL);

        ~DialogAboutQItom() {m_VersionString.clear();};

        QColor linkColor() { return m_linkColor; }
        void setLinkColor(const QColor &color);

        QColor textColor() { return m_textColor; }
        void setTextColor(const QColor &color);

    private:
        void styleTexts();

        Ui::DialogAboutQItom ui;
        QString m_VersionString;

        QColor m_textColor;
        QColor m_linkColor;

        QString m_aboutText;
        QString m_contributorsText;
        QString m_adddressText;
        QString m_addressText;

    public slots:


    private slots:
        void on_pushButtonCopy_clicked();
        void on_pushButton_close_clicked();

};

} //end namespace ito

#endif // DIALOGABOUTQITOM
