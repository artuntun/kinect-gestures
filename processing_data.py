from collections import deque, namedtuple
import numpy as np

Coordenates = namedtuple('Coordenates', 'hand_left, elbow_left, elbow_right, hand_right, neck, spine')

class SkeletonFrame():
    """This class store data for each Skeletonframe"""

    def __init__(self, line):
        tokens = line.split()
        self.label = tokens[0]
        self.coordenates = Coordenates(np.array(tokens[1:4], dtype=np.float32),
                                        np.array(tokens[4:7], dtype=np.float32),
                                        np.array(tokens[7:10], dtype=np.float32),
                                        np.array(tokens[10:13], dtype=np.float32),
                                        np.array(tokens[13:16], dtype=np.float32),
                                        np.array(tokens[16:19], dtype=np.float32))

def load_skeleton(file):
    with open(file, 'r') as f:
        essay_list = list(f)

    skeleton_queue = deque()
    trial_queue = deque()

    for line in essay_list:
        if line == "\r\n":
            trial_queue.append(skeleton_queue)
            skeleton_queue = deque()
        else:
            skeleton_queue.append(SkeletonFrame(line))
    return trial_queue

trial_queue = load_skeleton("skeltonData.txt")

#GET CENTROID of last essay
essay = trial_queue[-1]
sum = np.array([0.0,0.0,0.0], dtype=np.float32)
for i,ske in enumerate(essay):
    sum = sum + ske.coordenates.spine
    centroid = sum / (i+1)
    #print "{} : {}".format(i,centroid)

print "WAS: {}".format(essay[-1].coordenates.spine)
# substracting centroid from all cordenates of the essay
"""for o,ske_frame in enumerate(essay):
    for u,point in enumerate(ske_frame.coordenates):
        was = essay[o].coordenates[u]
        essay[o].coordenates[u] = point - centroid
        now = essay[o].coordenates[u]
        print " {} - {} = {}".format(was,centroid,now)"""

essay[-1].coordenates[-1] = np.array([1.2,2,3.1], dtype=np.float32)

print "NOW IS: {}".format(essay[-1].coordenates.spine)
