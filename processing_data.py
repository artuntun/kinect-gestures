from collections import deque

class SkeletonFrame():
    """This class store data for each Skeletonframe"""

    def __init__(self, line):
        tokens = line.split()
        self.label = tokens[0]
        self.handLeft = tokens[1:4]
        self.elbowLeft = tokens[4:7]
        self.elbowRight = tokens[7:10]
        self.handRight = tokens[10:13]
        self.neck = tokens[13:16]
        self.spine = tokens[16:19]

def load_skeleton(file):
    f = open(file,"r")
    skeleton_queue = deque()
    trial_queue = deque()

    essay_list = list(f)

    for line in essay_list:
        if line == "\r\n":
            trial_queue.append(skeleton_queue)
            skeleton_queue = deque()
        else:
            skeleton_queue.append(SkeletonFrame(line))

    return trial_queue

trial_queue = load_skeleton("skeltonData.txt")


skeleton = trial_queue.pop().pop()

print skeleton.spine

skeleton2 = trial_queue.pop().pop()

print skeleton2.spine
