/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut für Technische Optik (ITO),
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

#ifndef GUIHELPER
#define GUIHELPER

#include "../global.h"


namespace ito {

class GuiHelper
{
public:
    static int getScreenLogicalDpi(const QPoint *pos = nullptr);
    static float screenDpiFactor(const QPoint* pos = nullptr); /*!< returns a factor [1,..) by which user interface components
                                       can be scaled due to high resolution screens. A factor of 1.0
                                       is related to the default 96dpi screen. */

    //! converts a font weight, read from a settings or theme file, to the scale of the current Qt version.
    /*!
        Qt5 uses font weights in the range [0, 99] (normal: 50, bold: 75), Qt6 uses [1, 1000]
        (normal: 400, bold: 700). Settings and editor themes can have been written by both
        versions. Weights of the other scale are converted, such that the font is rendered as
        intended (e.g. not hairline-thin in Qt6 for a weight of 50).

        \param weight is the weight as stored in the file
        \return the weight in the scale of the current Qt version
    */
    static int normalizeFontWeight(int weight);

private:
    static int dpi;
    static QString highDPIFile;
    static QList<int> screensDPI;
};

} // namespace ito

#endif // GUIHELPER
