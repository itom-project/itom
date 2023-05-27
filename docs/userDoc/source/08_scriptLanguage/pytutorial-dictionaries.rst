

.. include:: ../include/global.inc

Dictionaries, Lists and Tuples
===============================
.. moduleauthor:: PSchau
.. sectionauthor:: PSchau




Dictionaries
--------------

One of Python's built-in datatypes is the dictionary, which defines one-to-one relationships between keys and values.


Defining Dictionaries
^^^^^^^^^^^^^^^^^^^^^^

It is best to think of a dictionary as an unordered set of *key: value* pairs, with the requirement that the keys are unique (within one dictionary). A pair of braces creates an empty dictionary: ``{}``. Placing a comma-separated list of key:value pairs within the braces adds initial key:value pairs to the dictionary; this is also the way dictionaries are written on output.

.. code-block:: python
    :linenos:

    d = {"lens":"zoom", "software":"itom"}
    print(d)

    # 'lens' is a key and its associated value is 'zoom'. It can be referenced by d["lens"].
    print(d["lens"])

    # 'software' is a key and its associated value is 'itom'. It can be referenced by d["software"].
    print(d["software"])

    # You can get values by key, but you can't get keys by value.
    # So d["lens"] is 'zoom', but d["zoom"] raises an exception, because 'zoom' is not a key.
    print(d["zoom"])

::

    {'lens': 'zoom', 'software': 'itom'}
    zoom
    itom
    Traceback (most recent call last):
      File "..", line ?, in <module>
    KeyError: 'zoom'


Modifying Dictionaries
^^^^^^^^^^^^^^^^^^^^^^^^

You can not have duplicate keys in a dictionary. Assigning a value to an existing key will wipe out the old value. You can add new key-value pairs at any time. This syntax is identical to modifying existing values. Dictionaries have no concept of order among elements.

.. code-block:: python
    :linenos:

    d = {"lens":"zoom", "software":"itom"}
    print(d)

    d["software"] = "itom"
    print(d)

    d["institute"] = "ITO"
    print(d)

    # Dictionary keys can be modified. The old value is simply replaced with a new one.
    d["institute"] = "none"
    print(d)

    # Dictionary keys are case-sensitive. Here, a new key-value pair will be created.
    d["Institute"] = "another"
    print(d)

::

    {'software': 'itom', 'lens': 'zoom'}
    {'software': 'itom', 'lens': 'zoom'}
    {'software': 'itom', 'institute': 'ITO', 'lens': 'zoom'}
    {'software': 'itom', 'institute': 'none', 'lens': 'zoom'}
    {'Institute': 'another', 'software': 'itom', 'institute': 'none', 'lens': 'zoom'}

Dictionaries aren't just for strings. Dictionary values can be any datatype, including strings, integers, objects, or even other dictionaries. Within a single dictionary, the values don't all need to be the same type; you can mix and match as needed. Dictionary keys are more restricted, but they can be strings, integers, and a few other types. You can also mix and match key datatypes within a dictionary.

.. code-block:: python
    :linenos:

    d = {'lens': 'zoom', 'institute': 'ITO', 'software': 'itom'}
    print(d)

    d["version"] = 3
    print(d)

    d[42] = "douglas"
    print(d)

::

    {'lens': 'zoom', 'institute': 'ITO', 'software': 'itom', 'version': 3}
    {'lens': 'zoom', 'institute': 'ITO', 'software': 'itom', 42: 'douglas', 'version': 3}


Deleting Items From Dictionaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code-block:: python
    :linenos:

    d = {'lens': 'zoom', 'institute': 'ITO', 'software': 'itom', 42: 'douglas', 'retrycount': 3}
    print(d)

    # delete individual items from a dictionary by key
    del d[42]
    print(d)

    # delete all items from a dictionary
    # Note that the set of empty curly braces in the output signifies a dictionary without any items.
    d.clear()
    print(d)

::

    {'lens': 'zoom', 'institute': 'ITO', 'software': 'itom', 42: 'douglas', 'retrycount': 3}
    {'lens': 'zoom', 'institute': 'ITO', 'software': 'itom', 'retrycount': 3}
    {}


Lists
------

Lists are Python's workhorse datatype. Variables can be named anything, and Python keeps track of the datatype internally.


Defining Lists
^^^^^^^^^^^^^^

A list is an ordered set of elements enclosed in square brackets. It can be used like a zero-based array.

.. code-block:: python
    :linenos:

    # A list of five elements is defined. Note that they retain their original order.
    li = ["a", "b", "zoom", "z", "example"]
    print(li)

    # The first element of any non-empty list is always li[0]
    print(li[0])

    # The last element of this five-element list is li[4].
    print(li[4])

::

    ['a', 'b', 'zoom', 'z', 'example']
    a
    example


Negative List Indices
^^^^^^^^^^^^^^^^^^^^^^

A negative index accesses elements from the end of the list counting backwards. The last element of any non-empty list is always ``li[-1]``. If the negative index is confusing to you, think of it this way: ``li[-n] == li[len(li) - n]``.

.. code-block:: python
    :linenos:

    li = ["a", "b", "zoom", "z", "example"]
    print(li)
    print(li[-1])

    # In this list, li[-3] == li[5 - 3] == li[2]
    print(li[-3])

::

    ['a', 'b', 'zoom', 'z', 'example']
    'example'
    'zoom'


Slicing a List
^^^^^^^^^^^^^^

You can get a subset of a list, called a `slice` by specifying two indices. The return value is a new list containing all the elements of the list, in order, starting with the first slice index up to but not including the second slice index.

If it helps, you can think of it this way: reading the list from left to right, the first slice index specifies the first element you want, and the second slice index specifies the first element you don't want. The return value is everything in between.

.. code-block:: python
    :linenos:

    li = ["a", "b", "zoom", "z", "example"]
    print(li)

    # A slice of li[1] up to but not including li[3] will be created
    print(li[1:3])

    # Slicing works if one or both of the slice indices is negative.
    print(li[1:-1])

    # Lists are zero-based, so li[0:3] returns the first three elements of the list,
    # starting at li[0], up to but not including li[3].
    print(li[0:3])

    # If the left slice index is 0, you can leave it out, and 0 is implied. So li[:3]
    # is the same as li[0:3]
    print(li[:3])

    # Similarly, if the right slice index is the length of the list, you can leave it
    # out. So li[3:] is the same as li[3:5], because this list has five elements.
    print(li[3:])

    # If both slice indices are left out, all elements of the list are included.
    # li[:] is shorthand for making a complete copy of a list.
    print(li[:])

::

    ['a', 'b', 'zoom', 'z', 'example']
    ['b', 'zoom']
    ['b', 'zoom', 'z']
    ['a', 'b', 'zoom']
    ['a', 'b', 'zoom']
    ['z', 'example']
    ['a', 'b', 'zoom', 'z', 'example']


Adding Elements to Lists
^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code-block:: python
    :linenos:

    li = ["a", "b", "zoom", "z", "example"]
    print(li)

    # Single elements can be added to the end of the list with append
    li.append("new")
    print(li)

    # insert inserts a single element into a list. The numeric argument is the index of the
    # first element that gets bumped out of position.
    li.insert(2, "new")
    print(li)

    # Lists can be concatenated with extend. Note that you do not call extend with multiple
    # arguments; you call it with one argument, a list. In this case, that list has two elements.
    li.extend(["two", "elements"])
    print(li)

::

    ['a', 'b', 'zoom', 'z', 'example']
    ['a', 'b', 'zoom', 'z', 'example', 'new']
    ['a', 'b', 'new', 'zoom', 'z', 'example', 'new']
    ['a', 'b', 'new', 'zoom', 'z', 'example', 'new', 'two', 'elements']


Difference between append und extend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lists have two methods, :py:func:`extend` and :py:func:`append`, that look like they do the same thing, but are in fact completely different. :py:func:`extend` takes a single argument, which is always a list, and adds each of the elements of that list to the original list. On the other hand, :py:func:`append` takes one argument, which can be any data type, and simply adds it to the end of the list.

.. code-block:: python
    :linenos:

    # extend method
    li = ['a', 'b', 'c']

    # li is extended with a list of another three elements ('d', 'e', and 'f'), so you
    # now have a list of six elements.
    li.extend(['d', 'e', 'f'])
    print(li)
    print(len(li))
    print(li[-1])

    #append method
    li = ['a', 'b', 'c']

    # append method is called with a single argument, which is a list of three elements
    # Now the list contains four elements because the last element appended is itself a
    # list. Lists can contain any type of data, including other lists.
    li.append(['d', 'e', 'f'])
    print(li)
    print(len(li))
    print(li[-1])

::

    ['a', 'b', 'c', 'd', 'e', 'f']
    6
    f
    ['a', 'b', 'c', ['d', 'e', 'f']]
    4
    ['d', 'e', 'f']


Searching Lists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code-block:: python
    :linenos:

    li = ['a', 'b', 'new', 'zoom', 'z', 'example', 'new', 'two', 'elements']
    print(li)

    # index finds (only) the first occurrence of a value in the list and returns the index
    print(li.index("example"))
    print(li.index("new"))

    # If the value is not found in the list, Python raises an exception.
    # To test whether a value is in the list, use in, which returns True if the value is
    # found or False if it is not.
    print(li.index("c"))

::

    5
    2
    Traceback (most recent call last):
      File "...", line ?, in ?
    ValueError: 'c' is not in list


Deleting List Elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code-block:: python
    :linenos:

    li = ['a', 'b', 'new', 'zoom', 'z', 'example', 'new', 'two', 'elements']
    print(li)

    # remove (only) the first occurrence of a value from a list.
    li.remove("z")
    print(li)

    # removes only the first occurrence of a value
    li.remove("new")
    print(li)

    # pop removes the last element of the list, and it returns the value that it removed
    li.pop()
    print(li)

    # If the value is not found in the list, Python raises an exception.
    li.remove("c")

::

    ['a', 'b', 'new', 'zoom', 'z', 'example', 'new', 'two', 'elements']
    ['a', 'b', 'new', 'zoom', 'example', 'new', 'two', 'elements']
    ['a', 'b', 'zoom', 'example', 'new', 'two', 'elements']
    ['a', 'b', 'zoom', 'example', 'new', 'two']
    Traceback (most recent call last):
      File "...", line ?, in ?
    ValueError: list.remove(x): x not in list


Using List Operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lists can also be concatenated with the ``+`` operator. ``list = list + otherlist`` has the same result as ``list.extend(otherlist)``. But the ``+`` operator returns a new (concatenated) list as a value, whereas extend only alters an existing list. This means that extend is faster, especially for large lists.

.. code-block:: python
    :linenos:

    li = ['a', 'b', 'zoom']
    li = li + ['example', 'new']
    print(li)

    # li += ['two'] is equivalent to li.extend(['two'])
    li += ['two']
    print(li)

    # The * operator works on lists as a repeater: li = [1, 2] * 3 is equivalent to
    # li = [1, 2] + [1, 2] + [1, 2], which concatenates three lists into one.
    li = [1, 2] * 3
    print(li)

::

    ['a', 'b', 'zoom', 'example', 'new']
    ['a', 'b', 'zoom', 'example', 'new', 'two']
    [1, 2, 1, 2, 1, 2]


Tuples
---------

A tuple is an immutable list and can not be changed in any way once it is created. A tuple is defined in the same way as a list, except that the whole set of elements is enclosed in parentheses instead of square brackets. The elements of a tuple have a defined order and the indices are zero-based, just like a list.

Tuples are faster than lists. If you're defining a constant set of values and all you're ever going to do with it is iterate through it, use a tuple instead of a list. Tuples can be converted into lists, and vice-versa. The built-in tuple function takes a list and returns a tuple with the same elements, and the list function takes a tuple and returns a list.

.. code-block:: python
    :linenos:

    t = ("a", "b", "mpilgrim", "z", "example")
    print(t)

    # The first element of a non-empty tuple is always t[0]
    print(t[0])

    # Negative indices count from the end of the tuple, just as with a list.
    print(t[-1])

    # Slicing works too, just like a list.
    print(t[1:3])

::

    ('a', 'b', 'mpilgrim', 'z', 'example')
    a
    example
    ('b', 'mpilgrim')

Keep in mind that tuples have not methods.

.. code-block:: python
    :linenos:

    t = ('a', 'b', 'mpilgrim', 'z', 'example')

    # all following examples cause errors
    # t.append("new")
    # t.remove("z")
    # t.index("example")

    # You can use "in" to see if an element exists in the tuple.
    "z" in t
