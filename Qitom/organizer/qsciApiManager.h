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

#ifndef QSCIAPIMANAGER_H
#define QSCIAPIMANAGER_H

// Under Windows, define QSCINTILLA_MAKE_DLL to create a Scintilla DLL, or
// define QSCINTILLA_DLL to link against a Scintilla DLL, or define neither
// to either build or link against a static Scintilla library.
//!< this text is coming from qsciglobal.h
#define QSCINTILLA_DLL  //http://www.riverbankcomputing.com/pipermail/qscintilla/2007-March/000034.html

#include <Qsci/qsciscintilla.h>
#include <Qsci/qscilexerpython.h>
#include "Qsci/qsciapis.h"

#include <qstringlist.h>
#include <qdatetime.h>

namespace ito
{

    class QsciApiManager : public QObject
    {
        Q_OBJECT

        public:
            static QsciApiManager * getInstance(void);

            inline QsciAPIs *getQsciAPIs() const { qDebug("return m_pApi"); return m_pApi; }
            inline bool isPreparing() const { return m_isPreparing; }

            int updateAPI(QStringList files, bool forcePreparation = false);

            struct APIFileInfo
            {
                APIFileInfo() : absoluteFilename(""), checksum(0), exists(0) {}
                public:
                    QString absoluteFilename;
                    quint16 checksum;
                    bool exists;
                    QDateTime lastModified;

                    //sorting depends on filename only
                    bool operator < (const APIFileInfo &rhs) const
                    {
                        return absoluteFilename < rhs.absoluteFilename;
                    }
            };

        protected:

        private:
            QsciApiManager(void);
            inline QsciApiManager(QsciApiManager  &/*copyConstr*/) : QObject(), m_isPreparing(0) {}
            ~QsciApiManager(void);

            QsciAPIs* m_pApi;
            QsciLexerPython* m_qSciLex;

            bool m_isPreparing;
            QList<APIFileInfo> m_preparingAPIFiles;
            QString m_preparingFileInfo;

            bool m_loaded;

            static QsciApiManager *m_pQsciApiManager;

            //!< singleton nach: http://www.oop-trainer.de/Themen/Singleton.html
            class QsciApiSingleton
            {
                public:
                    ~QsciApiSingleton()
                    {
                        #pragma omp critical
                        {
                            if( QsciApiManager::m_pQsciApiManager != NULL)
                            {
                                delete QsciApiManager::m_pQsciApiManager;
                                QsciApiManager::m_pQsciApiManager = NULL;
                            }
                        }
                    }
            };
            friend class QsciApiSingleton;

        signals:

        public slots:

        private slots:
            void apiPreparationFinished();
            void apiPreparationCancelled();
            void apiPreparationStarted();
    };
} //namespace ito

#endif
