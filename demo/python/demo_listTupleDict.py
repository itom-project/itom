"""Lists, tuples, dictionaries
==============================

Lists
"""
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPython.png'

list1 = ["physics", "chemistry", 1997, 2000]
list2 = [1, 2, 3, 4, 5]
list3 = ["a", "b", "c", "d"]

print("list1[0]: ", list1[0])
print("list2[1:5]: ", list2[1:5])

###############################################################################
# Updating lists
list1[2] = 2001
print("New value available at index 2 : ", list1[2])

###############################################################################
# Delete list item
del list1[2]
print("After deleting value at index 2 : ", list1)

###############################################################################
# List operations
len(list1)  # length of list
list1 + list2  # concatenate list
["ho"] * 4  # repetition
2000 in list1  # membership
for obj in list1:  # iterations
    print(obj)

###############################################################################
# List functions, methods
list1.append("newVal")  # append value
list1.append(2000)
list1.count(2000)  # count how many times object occurs in list
list1.index(2000)  # lowest index in list object appears
list1.insert(3, 2022)  # insert object into list at index#
list1.pop()  # remove last object from list
list1.remove(2022)  # remove object from list
list1.reverse()  # reverse objects in list
list2.sort()  # sorts objects in list. Works only if list contains object of same dtype

###############################################################################
# Tuples
tup1 = ("physics", "chemistry", 1997, 2000)
tup2 = (1, 2, 3, 4, 5, 6, 7)
print("tup1[0]: ", tup1[0])
print("tup2[1:5]: ", tup2[1:5])

###############################################################################
# Updating tuples
tup3 = tup1 + tup2
print(tup3)

###############################################################################
# Tuples operations
len(tup1)  # length of list
tup1 + tup2  # concatenate list
("ho") * 4  # repetition
2000 in tup1  # membership
for obj in tup1:  # iterations
    print(obj)

###############################################################################
# Dictionary
dict1 = {"Name": "Zara", "Age": 7, "Class": "First"}
print("dict1['Name']: ", dict1["Name"])

###############################################################################
# Updating dictionary
dict1["Age"] = 8  # update existing entry
dict1["School"] = "DPS School"  # Add new entry

print("dict1['Age']: ", dict1["Age"])
print("dict1['School']: ", dict1["School"])

###############################################################################
# Delete elements
del dict1["Name"]  # remove entry with key 'Name'
print("dict1['Age']: ", dict1["Age"])
print("dict1['School']: ", dict1["School"])
dict1.clear()  # remove all entries in dict
del dict1  # delete entire dictionary

###############################################################################
# Dictionary functions, methods
dict1 = {"Name": "Zara", "Age": 7, "Class": "First"}
dict1.copy()  # returns shallow copy of dict
dict.fromkeys(dict1)  # new dict with keys from sequence and values set to value
dict1.items()  # returns list of dict (key, value) tuple pairs
dict1.keys()  # return list of keys
for key, value in dict1.items():  # iterations
    print(key, value)
