"""Thread pool executor
=======================

``Asyncio``/``concurrent`` heavily changed from python ``3.4`` to ``3.7``, better read the docs
and do some tutorials. Asyncio is preferred over plain concurrent module."""

import concurrent.futures
import urllib.request
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoThreadPool.png'

URLS = [
    "http://www.foxnews.com/",
    "http://www.cnn.com/",
    "http://europe.wsj.com/",
    "http://www.bbc.co.uk/",
    "http://some-made-up-domain.com/",
]


###############################################################################
# Retrieve a single page and report the url and contents
def load_url(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()


###############################################################################
# We can use a with statement to ensure threads are cleaned up promptly
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Start the load operations and mark each future with its URL
    future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print(f"{url!r} generated an exception: {exc}")
        else:
            print("%r page is %d bytes" % (url, len(data)))
