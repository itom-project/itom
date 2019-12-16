'''this is a modified example from the python documentation.
The only difference is the set_executable section at the start.
Please notice that you cannot use methods from the itom module
in any worker thread
Alternative approaches for multiprocessing are 
python threading module and asyncio. Or use subprocess...'''
from multiprocessing import Pool, TimeoutError
import multiprocessing
import time
import os



def f(x):
    print("return x*x with x = ", x)
    return x*x

if __name__ == '__main__':
    pythonPath = ui.getOpenFileName("Set path of python.exe", "C:/itom/3rdParty/Python/python.exe", "Python Executable (*.exe)")

    if pythonPath and os.path.exists(pythonPath):
        #set the path of the python executable (required is python is embedded)
        multiprocessing.set_executable(pythonPath)

        with Pool(processes=4) as pool:

            # print "[0, 1, 4,..., 81]"
            print(pool.map(f, range(10)))

            # print same numbers in arbitrary order
            for i in pool.imap_unordered(f, range(10)):
                print(i)

            # evaluate "f(20)" asynchronously
            res = pool.apply_async(f, (20,))      # runs in *only* one process
            print(res.get(timeout=1))             # prints "400"

            # evaluate "os.getpid()" asynchronously
            res = pool.apply_async(os.getpid, ()) # runs in *only* one process
            print(res.get(timeout=1))             # prints the PID of that process

            # launching multiple evaluations asynchronously *may* use more processes
            multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
            print([res.get(timeout=1) for res in multiple_results])

            # make a single worker sleep for 10 secs
            res = pool.apply_async(time.sleep, (10,))
            try:
                print(res.get(timeout=1))
            except TimeoutError:
                print("We lacked patience and got a multiprocessing.TimeoutError")

            print("For the moment, the pool remains available for more work")

        # exiting the 'with'-block has stopped the pool
        print("Now the pool is closed and no longer available")