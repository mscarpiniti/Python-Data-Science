# -*- coding: utf-8 -*-
"""
Example of showing folder contents.

Created on Sat Jan 18 11:22:07 2025

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""


directory = "C:/Notebooks/PyDS/"

directories = os.listdir(directory)
directories.sort()

for d in directories:
    path_directories = directory + d
    print(path_directories)
    file_list = os.listdir(path_directories)

	for f in file_list:
        file_name = path_directories + '/' + f
        print(file_name)
