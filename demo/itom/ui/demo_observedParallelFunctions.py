"""Observe parallel functions
=============================

This demo shows how to use a customized progressObserver to observer and
possibly cancel multiple 'complex' function calls, that are executed in 
parallel in four different background threads.

Each single function call has its own progressObserver (member observer of class
FunctionCall). Additionally, there is one major progressObserver, that tracks
the total state of all coomplex functions call (each lasting for 10 seconds).

In this demo script, we do not only use the pre-defined possibilities of
the class ``itom.progressObserver`` to show the current progress value and
text in a given ``itom.uiItem`` of the GUI, but we use the 
``itom.progressObserver.connect`` method to connect different callable python
methods to signals of the progressObserver such that customized actions
can be done.

The overall observer is set to a range between 0 and 100 (percent). Every of the four
parallel function calls can add up to 25% of the total progress. Once the
overall progress reaches 100%, the call is considered to be finished.

Using the cancellation features of each progressObserver, this demo further
shows, that it is both possible to only cancel single function calls or the
entire call to all (currently running) sub-functions.

This demo uses the Thread class of python. Since python does not allow
real concurrent calls within python itself, we do not have to handle race conditions.

Hint: Algorithm calls of itom algorithms will release the Python GIL during the
entire time of the function call, therefore other Python thread can work in the
meantime.
"""

from itomUi import ItomUi
from itom import progressObserver
from threading import Thread
from time import sleep
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoObserveParallelFunction.png'


class FunctionCall:
    """This class wraps GUI items, the thread and observer of one function call.
    """

    def __init__(self, label, progress, abortButton, overallObserver):
        """Constructor."""
        self.label = label
        self.progress = progress
        self.abortBtn = abortButton
        self.thread = None

        # create the local observer for one function call
        self.observer = progressObserver(
            progressBar=self.progress, progressMinimum=0, progressMaximum=100
        )

        # pass the overall observer
        self.overallObserver = overallObserver

        # make several connections to signals of some objects with methods of this class
        self.abortBtn.connect("clicked()", self.on_abortButton_clicked)
        self.observer.connect(
            "progressValueChanged(int)", self.on_progressValue_changed
        )
        self.observer.connect(
            "progressTextChanged(QString)", self.on_progressText_changed
        )
        self.observer.connect(
            "cancellationRequested()", self.on_observer_cancellationRequested
        )

        # True if the function is currently executed
        self.inProgress = False

        # Last reported progress value to the overallObserver [0, 25] in percent.
        self._lastReportedProgressValue = 0

    def reset(self):
        """Resets the relevant GUI items to an idle state."""
        self.label["text"] = "-"
        self.progress["value"] = 0
        self.progress["enabled"] = False
        self.abortBtn["enabled"] = False

    def on_abortButton_clicked(self):
        """Callback if the abort button of this function is clicked."""
        if self.observer:
            self.abortBtn["enabled"] = False
            # force the observer to cancel the running algorithm. The algorithm
            # has to regularily check for this request and terminate the algorithm
            # (with an exception set) as soon as possible.
            self.observer.requestCancellation()

    def start(self):
        """Start the complex function in a Python thread by executing self.run."""
        self.thread = Thread(target=self.run)
        self.thread.start()

    def cancel(self):
        """Method to request a cancellation of the algorithm call as public interface.
        
        This method is usually called if the global abort button is clicked."""
        self.observer.requestCancellation()

    def run(self):
        """Run method, executed in a thread.
        
        This method mainly starts the itom algorithm ``demoCancellationFunction``
        and passes the local observer to this function. If this observer should
        be requested to cancel, the algorithm will return with a RuntimeError.
        
        This exception is handled. At the end, the contribution to the global
        progress of this function is set to the maximum of 25% (Even in the case
        of a cancellation).
        """
        self.observer.reset()

        self.progress["value"] = 0
        self.progress["enabled"] = True
        self.abortBtn["enabled"] = True

        try:
            self.inProgress = True
            filter("demoCancellationFunction", _observer=self.observer)
            # self.on_progressValue_changed(100)
        except RuntimeError:
            # cancellation
            pass
        finally:
            # done or cancelled: report a full progress of 25%
            self.overallObserver.progressValue += max(
                0, 25 - self._lastReportedProgressValue
            )
            self._lastReportedProgressValue = 25
            self.abortBtn["enabled"] = False
            self.inProgress = False
            sleep(1)
            self.reset()

    def on_progressValue_changed(self, value):
        """Callback if the local observer reports a new progress value."""
        if self.inProgress:
            if value < 100:
                self.label["text"] = "%i/100" % value
            else:
                self.label["text"] = "done"

            self.overallObserver.progressValue += max(
                0, (value // 4) - self._lastReportedProgressValue
            )
            self._lastReportedProgressValue = value // 4

    def on_progressText_changed(self, text):
        """Callback if the local observer reports a new progress text.
        
        Hint: it makes no real sense to change the toolTip. It is just an example."""
        self.label["toolTip"] = text

    def on_observer_cancellationRequested(self):
        """Callback if a cancellation has been requested to the local observer."""
        if self.inProgress:
            self.label["text"] = "cancelled"
            self.progress["value"] = 0


class DemoObserver(ItomUi):
    """Main GUI class that provides functionality to run for complex algorithms in parallel."""

    def __init__(self):
        """Constructor."""
        ItomUi.__init__(self, "observedParallelFunctions.ui")

        self.sets = (
            []
        )  # sets of widgets in the GUI for each parallel function execution

        # this overallObserver gives 25% to each of the 4 parallel function calls.
        self.overallObserver = progressObserver(
            progressBar=self.gui.progressAll,
            progressMinimum=0,
            progressMaximum=100,
        )
        self.overallObserver.connect(
            "progressValueChanged(int)", self.overallProgressChanged
        )

        # Initialization of four FunctionCall objects for four parallel complex
        # function calls.
        for idx in range(1, 5):
            self.sets.append(
                FunctionCall(
                    self.gui.getChild("lblRun%i" % idx),
                    self.gui.getChild("progressRun%i" % idx),
                    self.gui.getChild("btnAbortRun%i" % idx),
                    self.overallObserver,
                )
            )

        # reset and hide all besides the start button
        for s in self.sets:
            s.reset()

        self.gui.btnAbort["enabled"] = False
        self.gui.progressAll["visible"] = False

    @ItomUi.autoslot("")
    def on_btnStart_clicked(self):
        """Auto-connected slot, called if the start button is clicked."""
        self.gui.btnAbort["enabled"] = True
        self.gui.btnStart["enabled"] = False
        self.gui.progressAll["visible"] = True

        # resets the overall
        self.overallObserver.reset()

        # start the 4 threads with a short delay
        for s in self.sets:
            s.start()
            sleep(0.1)

    @ItomUi.autoslot("")
    def on_btnAbort_clicked(self):
        """Informs the algorithm call to interrupt as soon as possible."""

        # cancels all four function calls
        for s in self.sets:
            s.cancel()

        self.gui.btnAbort["enabled"] = False
        self.gui.btnStart["enabled"] = True
        self.gui.progressAll["visible"] = False

    def overallProgressChanged(self, value):
        """Callback if the progressValueChanged signal of the overall progressObserver is emitted."""
        if value >= 100:
            # all done
            self.gui.btnAbort["enabled"] = False
            self.gui.btnStart["enabled"] = True
            self.gui.progressAll["visible"] = False


if __name__ == "__main__":
    demoObserverGui = DemoObserver()
    demoObserverGui.show()
