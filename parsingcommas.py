# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:33:19 2015

@author: arturo
"""

with open('skeltonData.txt', 'r') as data:
  plaintext = data.read()

plaintext = plaintext.replace(',', '.')

