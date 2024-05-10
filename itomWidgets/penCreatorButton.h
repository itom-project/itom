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


#ifndef PENCREATORBUTTON_H
#define PENCREATORBUTTON_H

#include <QPushButton>
#include <qpen.h>

#include "commonWidgets.h"

class PenCreatorButtonPrivate;

class ITOMWIDGETS_EXPORT PenCreatorButton : public QPushButton
{
     Q_OBJECT

     Q_PROPERTY(QPen pen READ getPen WRITE setPen)
     Q_PROPERTY(bool editableColor READ getColorState WRITE setColorState DESIGNABLE true)

public:
    explicit PenCreatorButton(QWidget* parent = 0);
    explicit PenCreatorButton(QPen pen, QWidget* parent = 0 );
    ~PenCreatorButton();

    QSize sizeHint() const;
    QPen getPen() const;
    bool getColorState() const;

protected:
    virtual void paintEvent(QPaintEvent* event);
    void changePen();

    QScopedPointer<PenCreatorButtonPrivate> d_ptr;
protected slots:
    void onToggled(bool change = true);
public slots:
    ///
    ///  Set a new current pen without opening a dialog
    void setPen(const QPen &pen);
    void setColorState(const bool &val);
private:

       Q_DECLARE_PRIVATE(PenCreatorButton);
       Q_DISABLE_COPY(PenCreatorButton);
   signals:
       void colorStateChanged(bool state);

};






#endif
