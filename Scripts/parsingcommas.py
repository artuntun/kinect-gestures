# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:33:19 2015

@author: arturo
"""
print "starting parsing"

with open('skeletonData.txt', 'r') as data:
  plaintext = data.read()

plaintext = plaintext.replace(',', '.')

print "Finish parsing"
