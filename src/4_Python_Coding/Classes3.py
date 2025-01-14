# -*- coding: utf-8 -*-
"""
Example of inheritance

Created on Tue Dec 31 12:35:54 2024

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

class Person:
  def __init__(self, fname, lname):
    self.firstname = fname
    self.lastname = lname

  def printname(self):
    print(self.firstname, self.lastname)


class Student(Person):
  def __init__(self, fname, lname, year):
    super().__init__(fname, lname)
    self.graduationyear = year

  def welcome(self):
    print("Welcome", self.firstname, self.lastname, "to the class of", self.graduationyear)


person1 = Person('Michele', 'Scarpiniti')
person2 = Student('Mario', 'Rossi', 2020)

person1.printname()
person2.welcome()
