# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:28:47 2015

@author: arturo
"""
import nltk

class SkeletonFrame():
    """This class store data for each Skeletonframe"""

    def __init__(self, hRight, eRight, hLeft, eLeft, neck, spine, label):
        self.handRight = hRight
        self.handLeft = hLeft
        self.elbowRight = eRight
        self.elbowLeft = eLeft
        self.neck = neck
        self.spine = spine
        self.label = label
    
    def __init__(self):
        self.label="none"

    def showData(self):
        print(self.elbowLeft)
        print(self.label)

print "Hola"
f = open("skeltonData.txt","r")
line = " "
while(line!="\n"):
    line = f.readline()
    tokens = nltk.word_tokenize(line)
    for x in tokens:
        print x
    

#print skeleton.label

