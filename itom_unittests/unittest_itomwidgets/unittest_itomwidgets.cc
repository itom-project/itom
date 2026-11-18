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

#include <QApplication>
#include <QByteArray>
#include <QString>

#include <cstdlib>
#include <cstring>

namespace {

QtMessageHandler defaultMessageHandler = nullptr;

//! drops the 'This plugin does not support ...' warnings of the offscreen platform plugin.
void messageHandler(QtMsgType type, const QMessageLogContext& context, const QString& msg)
{
    if (type == QtWarningMsg && msg.startsWith("This plugin does not support"))
    {
        return;
    }

    if (defaultMessageHandler)
    {
        defaultMessageHandler(type, context, msg);
    }
}

} // namespace

/*
 * Entry point of the itomWidgets GUI unittests.
 *
 * Widgets need a QApplication instance. If the environment variable QT_QPA_PLATFORM
 * is not set, the 'offscreen' platform plugin is used, such that the tests also run
 * on CI machines without a display. To watch the tests on screen, set
 * QT_QPA_PLATFORM to 'windows' (Windows) or 'xcb' (Linux) before starting.
 *
 * In contrast to the other itom unittest executables, this main function returns the
 * result of RUN_ALL_TESTS(). This lets CI pipelines and ctest detect failing tests.
 * The interactive 'pause' is only executed if the argument -pause is given.
 */
int main(int argc, char* argv[])
{
    if (qEnvironmentVariableIsEmpty("QT_QPA_PLATFORM"))
    {
        qputenv("QT_QPA_PLATFORM", QByteArray("offscreen"));
    }

    defaultMessageHandler = qInstallMessageHandler(messageHandler);

    // Use a fixed style, such that geometry dependent tests (e.g. handle positions)
    // behave the same on all platforms.
    QApplication::setStyle("Fusion");

    QApplication app(argc, argv);

    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();

    for (int i = 0; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "-pause") == 0)
        {
            const int ret = std::system("pause");
            Q_UNUSED(ret);
            break;
        }
    }

    return result;
}
