import threading
import time


def worker():
    """thread worker function"""
    print("worker start")
    time.sleep(2)
    print("worker end")
    return


threads = []
for i in range(5):
    t = threading.Thread(target=worker)
    threads.append(t)
    t.start()
