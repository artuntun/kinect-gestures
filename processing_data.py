from collections import deque
import numpy as np

class SkeletonFrame():
    """This class store data for each Skeletonframe"""

    def __init__(self, line):
        tokens = line.split()
        self.label = tokens[0]
        self.hand_left = np.array(tokens[1:4], dtype=np.float32)
        self.elbow_left = np.array(tokens[4:7], dtype=np.float32)
        self.elbow_right= np.array(tokens[7:10], dtype=np.float32)
        self.hand_right = np.array(tokens[10:13], dtype=np.float32)
        self.neck = np.array(tokens[13:16], dtype=np.float32)
        self.spine = np.array(tokens[16:19], dtype=np.float32)

# Read Data from file and generate a list of essays
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
    sum = sum + ske.spine
    centroid = sum / (i+1)

print essay[-1].spine
# substracting centroid from all cordenates of the essay
for o,ske_frame in enumerate(essay):
    essay[o].spine = essay[o].spine - centroid
    essay[o].hand_left = essay[o].hand_left - centroid
    essay[o].hand_right = essay[o].hand_right - centroid
    essay[o].elbow_left = essay[o].elbow_left - centroid
    essay[o].elbow_right = essay[o].elbow_right - centroid
    essay[o].neck = essay[o].neck - centroid
print essay[-1].spine
print "DONE"
