/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2016, Institut fuer Technische Optik (ITO),
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

#ifndef MAIN_H
#define MAIN_H

#include <qapplication.h>
#include <qdebug.h>
#include "opencv/cv.h"

class QItomApplication : public QApplication
{
    Q_OBJECT
public:
    
    QItomApplication ( int & argc, char ** argv ) : QApplication(argc,argv) {}

    bool notify ( QObject * receiver, QEvent * event )
    {
        try
        {
            return QApplication::notify(receiver,event);
        }
        catch (cv::Exception &exc)
        {
            qWarning("Itom-Application has caught a cv::exception");
            qWarning() << (exc.err).c_str() << " from" << receiver->objectName();
            //qDebug() << "Itom-Application caught an exception from" <<  receiver->objectName() << "from event type" << event->type();
            qFatal("Exiting due to exception caught. OpenCV-Exception: %s", (exc.err).c_str());
        }
        catch(std::exception &exc)
        {
            qWarning("Itom-Application has caught an exception");
            qWarning() << "Message:" << exc.what() << " from" << receiver->objectName();
            qFatal("Exiting due to exception caught. Exception: %s", exc.what());
        }
        catch (...)
        {
            qWarning("Itom-Application has caught an exception");
            qDebug() << "Itom-Application caught an exception from" <<  receiver->objectName() << "from event type" << event->type();
            qFatal("Exiting due to exception caught");
            
        }
        return false;
    }
};

#endif
