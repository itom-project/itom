try:
    import matlab
except Exception as ex:
    print("itom is possibly compiled without Matlab support. This demo is not working")
    raise ex


from itom import *


def demo_matlab():
    session = matlab.MatlabSession()  # a Matlab console is opened
    # Matlab can only be properly loaded if the libraries libeng.dll and libmx.dll (or libeng.so and libmx.so under linux)
    # can be properly found in the PATH of the operating system. An itom x64 also requires a Matlab x64 version and vice-versa.
    # Re-login to your computer after having changed the PATH variable, if Qt is also contained in the PATH variable,
    # put the Matlab path after Qt since the bin folder of Matlab also contains old Qt libraries.
    # If the matlab libraries could be loaded but the session could not be started, also see this link (for Windows users):
    # http://de.mathworks.com/help/matlab/matlab_external/register-matlab-as-automation-server.html

    session.setString(
        "myString", "test"
    )  # creates the string variable 'myString' in the Matlab workspace having the value 'test'
    print(
        "myString:", session.getString("myString")
    )  # returns 'test' as answer in tiom
    session.setValue(
        "myArray", dataObject.randN([2, 3], "int16")
    )  # creates a 2x3 random matrix in Matlab (name: myArray)
    arr = session.getValue(
        "myArray"
    )  # returns the 2x3 array 'myArray' from Matlab as Numpy array
    print(arr)

    # read the current working directory of matlab
    session.run("curDir = cd")
    print(session.getString("curDir"))

    # run directly executes the command (as string). This is the same than typing this command into the command line of Matlab.
    # use this to also execute functions in Matlab. At first, send all required variables to the Matlab workspace, then execute a function
    # that uses these variables.

    del session  # closes the session and deletes the instance

    # session.close() only closes the session


if __name__ == "__main__":
    demo_matlab()
