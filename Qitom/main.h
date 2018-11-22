/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2018, Institut fuer Technische Optik (ITO),
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
#include <iostream>
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
            /*int type = event ? event->type() : -1;

            if (type == 68 || type == 74 || type == 69)
            {
                QString className =  receiver->metaObject()->className();
                if (className == "ito::ScriptDockWidget")
                {
                    int j = 1;
                }
            }*/
            return QApplication::notify(receiver,event);
        }
        catch (cv::Exception &exc)
        {
            QString name = QString("%1 (%2)").arg(receiver->objectName()).arg(receiver->metaObject()->className());
            qWarning("Itom-Application has caught a cv::exception");
            qWarning() << (exc.err).c_str() << " from" << name;
            //qDebug() << "Itom-Application caught an exception from" <<  receiver->objectName() << "from event type" << event->type();
#ifdef _DEBUG
            qFatal("Exiting due to exception caught. OpenCV-Exception: %s", (exc.err).c_str());
#endif
            std::cerr << "Itom-Application has caught a cv::exception: " << (exc.err).c_str() << " from: " << name.toLatin1().constData() << "\n" << std::endl;
        }
        catch(std::exception &exc)
        {
            QString name = QString("%1 (%2)").arg(receiver->objectName()).arg(receiver->metaObject()->className());
            qWarning("Itom-Application has caught a std::exception");
            qWarning() << "Message:" << exc.what() << " from" << name;
#ifdef _DEBUG
            qFatal("Exiting due to std::exception caught. Exception: %s", exc.what());
#endif
            std::cerr << "Itom-Application has caught a std::exception: " << exc.what() << " from: " << name.toLatin1().constData() << "\n" << std::endl;
        }
        catch (...)
        {
			int enumIdx = QEvent::staticMetaObject.indexOfEnumerator("Type");
			QMetaEnum me = QEvent::staticMetaObject.enumerator(enumIdx);
			QByteArray key = event ? me.valueToKeys(event->type()) : "";
            int type = event ? event->type() : -1;
            QString name = QString("%1 (%2)").arg(receiver->objectName()).arg(receiver->metaObject()->className());
            qWarning("Itom-Application has caught an unknown exception");
            qWarning() << "Itom-Application caught an unknown exception from" <<  name << "from event type" << type << " (" << key.constData() << ")";
#ifdef _DEBUG
            qFatal("Exiting due to exception caught");
#endif
            std::cerr << "Itom-Application caught an unknown exception from: " << name.toLatin1().constData() << " from event type " << type << " (" << key.constData() << ")" << "\n" << std::endl;
        }
        return false;
    }
};

#endif
