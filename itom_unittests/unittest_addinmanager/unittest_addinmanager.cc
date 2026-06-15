/* ********************************************************************
itom software
URL: http://www.uni-stuttgart.de/ito
Copyright (C) 2017, Institut fuer Technische Optik (ITO),
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

#include "gtest/gtest.h"
#include <iostream>

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv); // Initializing the google test.

    RUN_ALL_TESTS(); // To start Test check

    bool executedByGoogleTestAdapter = false;

    for (int i = 0; i < argc; ++i)
    {
        const char* arg = argv[i];

        if (strcmp(arg, "-googletestadapter") == 0)
        {
            executedByGoogleTestAdapter = true;
            break;
        }
    }

    if (!executedByGoogleTestAdapter)
    {
        // execute this only, if the unittest is directly executed.
        // If it is executed by the VS Google Test Adapter extension,
        // no keyboard interaction must be implemented.
        std::system("pause");
    }

    return 0;
}
