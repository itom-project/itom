"""Basic variables
==================

"""
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPython.png'

counter = 100  # An integer assignment
miles = 1000.0  # A floating point
name = "itom"  # A string
comp = 1 + 1j * 5  # complex

print(counter)
print(miles)
print(name)
print(comp)

###############################################################################
# Get type of variable
print(type(counter))
print(type(miles))
print(type(name))
print(type(comp))

###############################################################################
# List, Tuples, Dictionary
listValues = ["abcd", 786, 2.23, "john", 70.2]  # appending, deleting possible
tuplesValues = ("abcd", 786, 2.23, "john", 70.2)  # tuples cannot be changed
tinydict = {"name": "john", "code": 6734, "dept": "sales"}
