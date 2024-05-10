"""Worker thread
================

This example shows how to use the Python ``threading`` package.
"""

import threading
import time

# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoWorkerThread.png'


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
