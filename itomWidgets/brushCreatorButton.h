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


#ifndef BRUSHCREATORBUTTON_H
#define BRUSHCREATORBUTTON_H

#include <QPushButton>
#include <qbrush.h>

#include "commonWidgets.h"

class BrushCreatorButtonPrivate;

class ITOMWIDGETS_EXPORT BrushCreatorButton : public QPushButton
{
    Q_OBJECT

        Q_PROPERTY(QBrush brush READ getBrush WRITE setBrush)
        Q_PROPERTY(bool showAlphaChannel READ getShowAlphaChannel WRITE setShowAlphaChannel)

public:
    explicit BrushCreatorButton(QWidget* parent = 0);
    explicit BrushCreatorButton(QBrush brush, QWidget* parent = 0);
    ~BrushCreatorButton();

    QSize sizeHint() const;
    QBrush getBrush() const;
    bool getShowAlphaChannel() const;

protected:
    virtual void paintEvent(QPaintEvent* event);
    void changeBrush();

    QScopedPointer<BrushCreatorButtonPrivate> d_ptr;
protected slots:
    void onToggled(bool change = true);
public slots:
    ///
    ///  Set a new current pen without opening a dialog
    void setBrush(const QBrush &brush);

    void setShowAlphaChannel(bool showAlphaChannel);
private:

    Q_DECLARE_PRIVATE(BrushCreatorButton);
    Q_DISABLE_COPY(BrushCreatorButton);

};






#endif
