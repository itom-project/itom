from itomUi import ItomUi
from itom import progressObserver
from threading import Thread
from time import sleep
import itom

class FunctionCall:
    
    def __init__(self,
                 label,
                 progress,
                 abortButton,
                 overallObserver):
        """
        """
        self.label = label
        self.progress = progress
        self.abortBtn = abortButton
        self.thread = None
        self.observer = progressObserver(
            progressBar=self.progress,
            progressMinimum=0,
            progressMaximum=100)
        self.overallObserver = overallObserver
        
        self.abortBtn.connect("clicked()", self.on_abortButton_clicked)
        self.observer.connect("progressValueChanged(int)", self.on_progressValue_changed)
        self.observer.connect("progressTextChanged(QString)", self.on_progressText_changed)
        self.observer.connect("cancellationRequested()", self.on_observer_cancellationRequested)
        self.inProgress = False
    
    def reset(self):
        self.label["text"] = "-"
        self.progress["value"] = 0
        self.progress["enabled"] = False
        self.abortBtn["enabled"] = False
    
    def on_abortButton_clicked(self):
        if self.observer:
            self.abortBtn["enabled"] = False
            self.observer.requestCancellation()
    
    def start(self):
        self.thread = Thread(target=self.run)
        self.thread.start()
    
    def cancel(self):
        self.observer.requestCancellation()
    
    def run(self):
        
        self.observer.reset()
        
        self.progress["value"] = 0
        self.progress["enabled"] = True
        self.abortBtn["enabled"] = True
        
        try:
            self.inProgress = True
            filter("demoCancellationFunction", _observer=self.observer)
            #self.on_progressValue_changed(100)
        except RuntimeError:
            # cancellation
            pass
        finally:
            self.overallObserver.progressValue += 1  # done or cancelled: report it!
            self.abortBtn["enabled"] = False
            self.inProgress = False
            sleep(1)
            self.reset()
    
    def on_progressValue_changed(self, value):
        if self.inProgress:
            if value < 100:
                self.label["text"] = "%i/100" % value
            else:
                self.label["text"] = "done"
    
    def on_progressText_changed(self, text):
        self.label["toolTip"] = text
        
    def on_observer_cancellationRequested(self):
        if self.inProgress:
            self.label["text"] = "cancelled"
            self.progress["value"] = 0


class DemoObserver(ItomUi):
    
    def __init__(self):
        ItomUi.__init__(self, "observedParallelFunctions.ui")
        
        self.sets = []  # sets of widgets in the GUI for each parallel function execution
        self.overallObserver = progressObserver(progressMinimum=0, progressMaximum=4)
        self.overallObserver.connect("progressValueChanged(int)", self.overallProgressChanged)
        
        for idx in range(1, 5):
            self.sets.append(
                FunctionCall(self.gui.getChild("lblRun%i" % idx),
                             self.gui.getChild("progressRun%i" % idx),
                             self.gui.getChild("btnAbortRun%i" % idx),
                             self.overallObserver))
        
        # reset and hide all besides the start button
        for s in self.sets:
            s.reset()
        
        self.gui.btnAbort["enabled"] = False

    @ItomUi.autoslot("")
    def on_btnStart_clicked(self):
        self.gui.btnAbort["enabled"] = True
        self.gui.btnStart["enabled"] = False
        
        self.overallObserver.progressValue = 0
        
        # start the 4 threads with a short delay
        for s in self.sets:
            s.start()
            sleep(0.1)
        

    @ItomUi.autoslot("")
    def on_btnAbort_clicked(self):
        """Informs the algorithm call to interrupt as soon as possible."""
        for s in self.sets:
            s.cancel()
        
        self.gui.btnAbort["enabled"] = False
        self.gui.btnStart["enabled"] = True
    
    def overallProgressChanged(self, value):
        if value >= 4:
            # all done
            self.gui.btnAbort["enabled"] = False
            self.gui.btnStart["enabled"] = True


demoObserverGui = DemoObserver()
demoObserverGui.show()

