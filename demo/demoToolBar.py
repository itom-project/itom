from itom import *

#Create a new application class
class myToolBar():
    #Add a ini-function
    def __init__(self):
        '''
        Init class with the only member variable test
        '''
        self.test = 1
    
    #Add the delete function, which is called from the python-garbage-collector randomly after the class was deleted
    def __del__(self):
        '''
        clean up buttonbar entries after killing the class
        '''
        removeButton("amipola", "HelloWorld")
        
    #Add a new function to the class
    def printHelloWorld(self, test):
        '''
        Try to print Hello-World and return test
        '''
        print("Hello World")
        return test
        
    def __len__(self):
        return 42

def demo_toolbar():
    global app
    
    # Create a new instance of myToolBar with the name app
    app = myToolBar()

    #Add the button Hello with the function myToolBar.printHelloWorld(...) to the buttonbar amipola
    addButton("amipola","HelloWorld","res = app.printHelloWorld(True)", "icons_m/HV_on.png")

    #For Debug test with single step
    app.printHelloWorld(True)
if __name__ == "__main__":
    demo_toolbar()