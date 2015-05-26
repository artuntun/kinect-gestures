

import nltk
from collections import deque

class SkeletonFrame():
    """This class store data for each Skeletonframe"""

    def __init__(self, iden, hRight, eRight, hLeft, eLeft, neck, spine, label):
        self.handRight = hRight
        self.handLeft = hLeft
        self.elbowRight = eRight
        self.elbowLeft = eLeft
        self.neck = neck
        self.spine = spine
        self.label = label
        self.iden = iden

    def __init__(self):
        self.label="none"

    def __init__(self, line):

        tokens = nltk.word_tokenize(line)
        aux = 0
        buff = []
        for x in tokens:
            if aux == 0:
                self.label = x
            elif aux == 1:
                buff.append(x)
            elif aux == 2:
                buff.append(x)
            elif aux == 3:
                buff.append(x)
                self.handLeft = buff
                #Clearing the buffer
                buff = []
            elif aux == 4:
                buff.append(x)
            elif aux == 5:
                buff.append(x)
            elif aux == 6:
                buff.append(x)
                self.elbowLeft = buff
                #Clearing the buffer
                buff = []
            elif aux == 7:
                buff.append(x)
            elif aux == 8:
                buff.append(x)
            elif aux == 9:
                buff.append(x)
                self.elbowRight = buff
                #Clearing the buffer
                buff = []
            elif aux == 10:
                buff.append(x)
            elif aux == 11:
                buff.append(x)
            elif aux == 12:
                buff.append(x)
                self.handRight = buff
                #Clearing the buffer
                buff = []
            elif aux == 13:
                buff.append(x)
            elif aux == 14:
                buff.append(x)
            elif aux == 15:
                buff.append(x)
                self.neck = buff
                #Clearing the buffer
                buff = []
            elif aux == 16:
                buff.append(x)
            elif aux == 17:
                buff.append(x)
            elif aux == 18:
                buff.append(x)
                self.spine = buff
                #Clearing the buffer
                buff = []
            aux = aux + 1



f = open("skeltonData.txt","r")
skeleton_queue = deque([])

line = "2"
queue = deque(list(f))

while(line != "\r\n"):
    line = queue.popleft()
    if (line != "\r\n"):
        skeleton_queue.append(SkeletonFrame(line))

skeleton = skeleton_queue.popleft()
print skeleton.spine