/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2026, Institut für Technische Optik (ITO),
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

#include "gtest/gtest.h"

#include <qdir.h>
#include <qapplication.h>
#include <qstring.h>
#include <qtemporarydir.h>

namespace ito {
namespace test {
void setSettingsFilePath(const QString& path);
}
} // namespace ito

int main(int argc, char** argv)
{
    // the model returns file icons (QFileIconProvider), therefore a QApplication is required
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);

    // every test gets a fresh settings file in a temporary directory
    QTemporaryDir tempDir;
    ito::test::setSettingsFilePath(tempDir.path() + QDir::separator() + "itom.ini");

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
