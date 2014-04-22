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

#ifndef DIALOGABOUTQITOM
#define DIALOGABOUTQITOM

#include <QtGui>
#include <qdialog.h>
//#include <qpair>
//#include <qlist>

#include "ui_dialogAbout.h"

namespace ito
{

class DialogAboutQItom : public QDialog 
{
    Q_OBJECT

    public:
        DialogAboutQItom(const QMap<QString, QString> &versionMap);

        ~DialogAboutQItom() {m_VersionString.clear();};

    private:
        Ui::DialogAboutQItom ui;
        QString m_VersionString;

    public slots:


    private slots:
        void on_pushButtonCopy_clicked();
        void on_pushButton_close_clicked();

};

} //end namespace ito

#endif // DIALOGABOUTQITOM
