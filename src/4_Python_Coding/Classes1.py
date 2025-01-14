# -*- coding: utf-8 -*-
"""
Example of defining and using classes. As a specific example, we develop
a class for managing rational numbers.

See:
    - Kenneth A. Lambert, Fundamentals of Python: First Programs,
    2nd Edition, Course Technology Ptr, 2018.
    - Kenneth A. Lambert, Programmazione in Python, Seconda Edizione,
    APogeo, 2018.

Created on Sun Feb 19 13:33:23 2023

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""


class Rat(object):
    """Represents a rational number."""

    def __init__(self, numer, denom):
        """Constructor creates a number with the given numerator
        and denominator and reduces it to lowest terms."""
        self.numer = numer
        self.denom = denom
        self._reduce()

    def numerator(self):
        """Returns the numerator."""
        return self.numer

    def denominator(self):
        """Returns the denominator."""
        return self.denom

    def __str__(self):
        """Returns the string representation of the number."""
        return str(self.numer) + "/" + str(self.denom)

    def _reduce(self):
        """Helper to reduce the number to lowest terms."""
        divisor = self._gcd(self.numer, self.denom)
        self.numer = self.numer // divisor
        self.denom = self.denom // divisor

    def _gcd(self, a, b):
        """Euclid's algorithm for greatest common divisor."""
        (a, b) = (max(a, b), min(a, b))
        while b > 0:
            (a, b) = (b, a % b)
        return a

    # Methods for overloading

    def __add__(self, other):
        """Returns the sum of the numbers."""
        newNumer = self.numer * other.denom + other.numer * self.denom
        newDenom = self.denom * other.denom
        return Rat(newNumer, newDenom)

    def __lt__(self, other):
        """Returns self < other."""
        extremes = self.numer * other.denom
        means = other.numer * self.denom
        return extremes < means

    def __ge__(self, other):
        """Returns self < other."""
        extremes = self.numer * other.denom
        means = other.numer * self.denom
        return extremes >= means

    def __eq__(self, other):
        """Tests self and other for equality."""
        if self is other:
            return True
        elif type(self) != type(other):
            return False
        else:
            return self.numer == other.numer and \
                   self.denom == other.denom



# %% Using the class

onesix = Rat(1, 6)
print(onesix)

sixeight = Rat(6, 8)
print(sixeight)

add = onesix + sixeight
print(add)

print(onesix < sixeight)

print(onesix == sixeight)
