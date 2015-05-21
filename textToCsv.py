# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:10:13 2015

@author: arturo
"""

import csv

txt_file = r"skeletonData.txt"
csv_file = r"skeletonData.csv"

# use 'with' if the program isn't going to immediately terminate
# so you don't leave files open
# the 'b' is necessary on Windows
# it prevents \x1a, Ctrl-z, from ending the stream prematurely
# and also stops Python converting to / from different line terminators
# On other platforms, it has no effect
in_txt = csv.reader(open(txt_file, "rb"), delimiter = ' ')
out_csv = csv.writer(open(csv_file, 'wb'))

out_csv.writerows(in_txt)