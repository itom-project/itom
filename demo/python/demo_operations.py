"""Basic operations
================

Arithmetic operations
"""
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPython.png'

a = 21
b = 10
c = 0

c = a + b
print("line 1 - value of c is ", c)

c = a - b
print("line 2 - value of c is ", c)

c = a * b
print("line 3 - value of c is ", c)

c = a / b
print("line 4 - value of c is ", c)

c = a % b
print("line 5 - value of c is ", c)

a = 2
b = 3
c = a**b
print("line 6 - value of c is ", c)

a = 10
b = 5
c = a // b
print("line 7 - value of c is ", c)

###############################################################################
# Comparison operations
a = 21
b = 10
c = 0

if a == b:
    print("line 1 - a is equal to b")
else:
    print("line 1 - a is not equal to b")

if a != b:
    print("line 2 - a is not equal to b")
else:
    print("line 2 - a is equal to b")

if a != b:
    print("line 3 - a is not equal to b")
else:
    print("line 3 - a is equal to b")

if a < b:
    print("line 4 - a is less than b")
else:
    print("line 4 - a is not less than b")

if a > b:
    print("line 5 - a is greater than b")
else:
    print("line 5 - a is not greater than b")

a = 5
b = 20
if a <= b:
    print("line 6 - a is either less than or equal to  b")
else:
    print("line 6 - a is neither less than nor equal to  b")

if b >= a:
    print("line 7 - b is either greater than  or equal to b")
else:
    print("line 7 - b is neither greater than  nor equal to b")

###############################################################################
# Bitwise operations
a = 60  # 60 = 0011 1100
b = 13  # 13 = 0000 1101
c = 0

c = a & b
# 12 = 0000 1100
print("line 1 - Value of c is ", c)

c = a | b
# 61 = 0011 1101
print("line 2 - Value of c is ", c)

c = a ^ b
# 49 = 0011 0001
print("line 3 - Value of c is ", c)

c = ~a
# -61 = 1100 0011
print("line 4 - Value of c is ", c)

c = a << 2
# 240 = 1111 0000
print("line 5 - Value of c is ", c)

c = a >> 2
# 15 = 0000 1111
print("line 6 - Value of c is ", c)

###############################################################################
# Logical operations
(a and b) is True
(a or b) is True
not (a and b) is False

###############################################################################
# Membership operations
a = 10
b = 20
list = [1, 2, 3, 4, 5]

if a in list:
    print("line 1 - a is available in the given list")
else:
    print("line 1 - a is not available in the given list")

if b not in list:
    print("line 2 - b is not available in the given list")
else:
    print("line 2 - b is available in the given list")

a = 2
if a in list:
    print("line 3 - a is available in the given list")
else:
    print("line 3 - a is not available in the given list")

###############################################################################
# Identify operations
a = 20
b = 20

if a is b:
    print("line 1 - a and b have same identity")
else:
    print("line 1 - a and b do not have same identity")

if id(a) == id(b):
    print("line 2 - a and b have same identity")
else:
    print("line 2 - a and b do not have same identity")

b = 30
if a is b:
    print("line 3 - a and b have same identity")
else:
    print("line 3 - a and b do not have same identity")

if a is not b:
    print("line 4 - a and b do not have same identity")
else:
    print("line 4 - a and b have same identity")

###############################################################################
# Operators Precedence
a = 20
b = 10
c = 15
d = 5
e = 0

e = (a + b) * c / d  # ( 30 * 15 ) / 5
print("value of (a + b) * c / d is ", e)

e = ((a + b) * c) / d  # (30 * 15 ) / 5
print("value of ((a + b) * c) / d is ", e)

e = (a + b) * (c / d)
# (30) * (15/5)
print("value of (a + b) * (c / d) is ", e)

e = a + (b * c) / d
#  20 + (150/5)
print("value of a + (b * c) / d is ", e)
