# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:33:05 2015

@author: arturo
"""

f = open("skeltonData.txt","r")

line = "2"
print f.name
lista = list(f)

while(line != "\r\n"):
    line = lista.pop()
    if (line != "\r\n"):
        print line


