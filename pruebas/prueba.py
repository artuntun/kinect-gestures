# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:02:30 2015

@author: arturo
"""

import csv
with open('/home/arturo/TFG/pruebas/prueba.csv', 'rb') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
     for row in spamreader:
         for x in row:
             print x