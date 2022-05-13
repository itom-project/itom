"""Decisions, loops
================

Decision making
"""
# sphinx_gallery_thumbnail_path = '11_demos/_static/_thumb/demoPython.png'

var = 100
if var == 100:
    print("value of expression is 100")

###############################################################################
# While loop
count = 0
while count < 9:
    print("The count is: ", count)
    count = count + 1

###############################################################################
# While loop with else statement
count = 0
while count < 5:
    print(count, " is  less than 5")
    count = count + 1
else:
    print(count, " is not less than 5")

###############################################################################
# For loop
for letter in "Python":  # First Example
    print("Current Letter :", letter)

fruits = ["banana", "apple", "mango"]
for fruit in fruits:  # Second Example
    print("Current fruit :", fruit)

###############################################################################
# Iterating by sequence index
fruits = ["banana", "apple", "mango"]
for index in range(len(fruits)):
    print("Current fruit :", fruits[index])

###############################################################################
# For loop with else
for num in range(10, 20):  # to iterate between 10 to 20
    for idx in range(2, num):  # to iterate on the factors of the number
        if num % idx == 0:  # to determine the first factor
            jdx = num / idx  # to calculate the second factor
            print("%d equals %d * %d" % (num, idx, jdx))
            break  # to move to the next number, the #first FOR
    else:  # else part of the loop
        print(num, "is a prime number")
        break

###############################################################################
# Nested loops
val = 2
while val < 100:
    val2 = 2
    while val2 <= (val / val2):
        if not (val % val2):
            break
        val2 = val2 + 1
    if val2 > val / val2:
        print(val, " is prime")
    val = val + 1

###############################################################################
# Break statement
for letter in "Python":  # First Example
    if letter == "h":
        break
    print("Current Letter :", letter)

var = 10  # Second Example
while var > 0:
    print("Current variable value :", var)
    var = var - 1
    if var == 5:
        break

###############################################################################
# Continue statement
for letter in "Python":  # First Example
    if letter == "h":
        continue
    print("Current Letter :", letter)

var = 10  # Second Example
while var > 0:
    var = var - 1
    if var == 5:
        continue
    print("Current variable value :", var)

###############################################################################
# Pass statement
for letter in "Python":
    if letter == "h":
        pass
        print("This is pass block")
    print("Current Letter :", letter)
