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

#pragma once

#include <qdialog.h>
#include <qstring.h>

#include "../helper/IOHelper.h"

#include "ui_dialogScriptCharsetEncoding.h"

namespace ito
{

class DialogScriptCharsetEncoding : public QDialog
{
    Q_OBJECT

public:
    DialogScriptCharsetEncoding(const IOHelper::CharsetEncodingItem &currentEncoding, bool enableReload, QWidget *parent);
    ~DialogScriptCharsetEncoding() {}

    IOHelper::CharsetEncodingItem getSaveCharsetEncoding() const;

private:
    Ui::DialogCharsetEncoding ui;

private slots:
    void on_btnReloadFile_clicked();
    void on_btnAddPythonEncodingComment_clicked();

Q_SIGNALS:
    void addPythonEncodingComment(const QString &encoding);
    void reloadWithEncoding(const IOHelper::CharsetEncodingItem &item);
};

} //end namespace ito
