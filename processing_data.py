"""
PREGUNTAS SOBRE EL CODIGO:
1. la variables de los for ejemplo "i" se reinician cuando salen del for
2. check the center_coordenates  function
3. Tupla eliminado porque no es mutable. Alguna otra solucion?
"""
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

#Center all the coordenates at the origin
def center_coordenates(trial_queue):

    sum = np.array([0.0,0.0,0.0], dtype=np.float32)
    for p,essay_aux in enumerate(trial_queue):

        #GET CENTROID of the current essay
        for i,ske in enumerate(trial_queue[p]):
            sum = sum + ske.spine
            centroid = sum / (i+1)
        sum = [0.0,0.0,0.0]
        #substract CENTROID from all the coordenates in the current essay
        for o,ske_frame in enumerate(trial_queue[p]):
            trial_queue[p][o].spine = trial_queue[p][o].spine - centroid
            trial_queue[p][o].hand_left = trial_queue[p][o].hand_left - centroid
            trial_queue[p][o].hand_right = trial_queue[p][o].hand_right - centroid
            trial_queue[p][o].elbow_left = trial_queue[p][o].elbow_left - centroid
            trial_queue[p][o].elbow_right = trial_queue[p][o].elbow_right - centroid
            trial_queue[p][o].neck = trial_queue[p][o].neck - centroid

    return trial_queue

#Normalize all the coordenates
def normalize_coordenates(trial_queue):
    norm = 0.15
    for p,essay_aux in enumerate(trial_queue):

        #GET mean distance between neck and spine of the current essay
        sum = 0.0
        for i,ske in enumerate(trial_queue[p]):
            distance = np.linalg.norm(ske.neck - ske.spine)
            sum = sum + distance
            mean = sum / (i+1)

        normalizer = mean/norm
        #substract CENTROID from all the coordenates in the current essay
        for o,ske_frame in enumerate(trial_queue[p]):
            trial_queue[p][o].spine = trial_queue[p][o].spine/normalizer
            trial_queue[p][o].hand_left = trial_queue[p][o].hand_left/normalizer
            trial_queue[p][o].hand_right = trial_queue[p][o].hand_right/normalizer
            trial_queue[p][o].elbow_left = trial_queue[p][o].elbow_left/normalizer
            trial_queue[p][o].elbow_right = trial_queue[p][o].elbow_right/normalizer
            trial_queue[p][o].neck = trial_queue[p][o].neck/normalizer

    return trial_queue

def extract_attributes(trial_queue):
    attributes_queue = []
    n = 6
    for p,essay_aux in enumerate(trial_queue):
        a = len(essay_aux)
        div = int(a/n)
        select = 0
        buffer_queue = []
        for x in range(0,n):
            buffer_queue.append(essay_aux[select])
            select = select + div
        attributes_queue.append(buffer_queue)
    return attributes_queue

data_set = load_skeleton("skeltonData.txt")
set_centered = center_coordenates(data_set)
set_normalize = normalize_coordenates(set_centered)

attributes = extract_attributes(set_normalize)


for essay in attributes:
    for ske in essay:
        print ske.spine

print "DONE"
