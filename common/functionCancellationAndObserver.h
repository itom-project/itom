/* ********************************************************************
    itom software
    URL: http://www.uni-stuttgart.de/ito
    Copyright (C) 2020, Institut fuer Technische Optik (ITO),
    Universitaet Stuttgart, Germany

    This file is part of itom and its software development toolkit (SDK).

    itom is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public Licence as published by
    the Free Software Foundation; either version 2 of the Licence, or (at
    your option) any later version.

    In addition, as a special exception, the Institut fuer Technische
    Optik (ITO) gives you certain additional rights.
    These rights are described in the ITO LGPL Exception version 1.0,
    which can be found in the file LGPL_EXCEPTION.txt in this package.

    itom is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library
    General Public Licence for more details.

    You should have received a copy of the GNU Library General Public License
    along with itom. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************** */

#ifndef FUNCCANCELLATIONANDOBSERVER_H
#define FUNCCANCELLATIONANDOBSERVER_H

#include "commonGlobal.h"
#include <qscopedpointer.h>
#include <qobject.h>



#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC) //only moc this file in itomCommonQtLib but not in other libraries or executables linking against this itomCommonQtLib

namespace ito
{

class FunctionCancellationAndObserverPrivate; //forward declaration

/*!
*    \class FunctionCancellationAndObserver
*    \brief This class can be passed to a long running method (e.g. as QSharedPointer instance) for two reasons:

     1. The long running method can use the progress methods in order to signal a progress. These changes of the
        progress are then signalled using specific signals.
     2. The caller of the long running method can set a cancellation request. The method itself should then check
        for the existence of this request and stop its calculation. The method should then return with an error message.

    This method is fully thread-safe such that read and write methods for progress as well as cancellation status
    can be called from different threads.
*/
class ITOMCOMMONQT_EXPORT FunctionCancellationAndObserver : public QObject
{
    Q_OBJECT

public:
    enum CancellationReason
    {
        ReasonGeneral = 1,
        ReasonKeyboardInterrupt = 2 //if the python script execution has been terminated
    };

    //! constructor
    /*!
        A new FuncitonCancellationAndObserver object is created.

        Pass this object to a function (thread-safe) in order to signal a possible cancellation request and / or
        to let the function signal its current progress (value and / or text).

        The progress should always be reported in the given range [progressMinimum, progressMaximum]
    */
    FunctionCancellationAndObserver(int progressMinimum = 0, int progressMaximum = 100, QObject *parent = NULL);

    //! destructor (do not call directly, instead free the semaphore by ItomSharedSemaphore::deleteSemaphore
    /*
    \sa ItomSharedSemaphore::deleteSemaphore
    */
    virtual ~FunctionCancellationAndObserver();

    //! return true if a cancellation request has been signalled. Else false.
    bool isCancelled() const;

    //! return the cancellation reason (call this only if isCancelled returned true)
    CancellationReason cancellationReason();

    //! returns the minimum progress value
    int progressMinimum() const;

    //! returns the maximum progress value
    int progressMaximum() const;

    //! changes the current value of the progress.
    /*
        This method emits the progressValueChanged signal. It can for instance be connected
        with a 'setValue' slot of a QProgressBar.

        The value will be clipped to progressMinimum and progressMaximum.
    */
    void setProgressValue(int value);

    //! returns the current progress value
    int progressValue() const;

    //! changes the current text of the progress
    /*
        This method emits the progressTextChanged signal. It can for instance be connected
        with a 'setText' slot of a QLabel.

        The text should inform about the step, the long-running method is currently executing.
    */
    void setProgressText(const QString &text);

    //! returns the current text of the progress
    QString progressText();

Q_SIGNALS:
    void progressTextChanged(QString text);
    void progressValueChanged(int value);
    void cancellationRequested();
    void resetDone();

public Q_SLOTS:
    //! call this method (from any thread) to signal a cancellation request.
    /*
    It is the responsibility of the called function itself to regulariy check the isCancelled() method
    and stop the function call as soon as possible.

    The reason ReasonKeyboardInterrupt should only be set by the method PythonEngine::pythonInterruptExecutionThreadSafe
    to avoid nested errors. In this case, itom.filter will not set an exception, since the keyboardInterrupt exception
    is automatically raised.

    Emits the cancellationRequested signal.
    */
    void requestCancellation(CancellationReason reason = CancellationReason::ReasonGeneral);

    //! resets this object (e.g. emptys the current progress text, set the progress value to its minimum and resets the cancellation request)
    /*
    Emits progressTextChanged and progressValueChanged with the new values and then the resetDone signal.
    */
    void reset();

private:
    Q_DISABLE_COPY(FunctionCancellationAndObserver)

    QScopedPointer<FunctionCancellationAndObserverPrivate> d_ptr; //!> self-managed pointer to the private class container (deletes itself if d_ptr is destroyed)
    Q_DECLARE_PRIVATE(FunctionCancellationAndObserver);
};

} //end namespace ito

#endif //#if !defined(Q_MOC_RUN) || defined(ITOMCOMMONQT_MOC)

#endif
