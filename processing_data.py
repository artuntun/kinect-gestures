"""
PREGUNTAS SOBRE EL CODIGO:
1. la variables de los for ejemplo "i" se reinician cuando salen del for
2. check the center_coordenates  function
3. Tupla eliminado porque no es mutable. Alguna otra solucion?
4. Cammbiar lista por numpy arrays en extract attributes
"""
from collections import deque
import numpy as np
from sklearn.naive_bayes import GaussianNB

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

def load_skeleton(file):
    """Read Data from file and generate a list of essays"""

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

def center_normalize_coordenates(trial_queue):
    """Center all the coordenates at the origin and normalize with a factor of 0.15"""

    norm = 0.15 #Normalaize factor
    spine_sum = np.array([0.0,0.0,0.0], dtype=np.float32)
    for p,essay_aux in enumerate(trial_queue):

        #GET CENTROID and mean distance between neck and spine of the current essay
        distance_sum = 0.0
        for i,ske in enumerate(trial_queue[p]):
            distance = np.linalg.norm(ske.neck - ske.spine)
            distance_sum += distance
            spine_sum += ske.spine
            centroid = spine_sum / (i+1)
            mean = distance_sum / (i+1)
        normalizer = mean/norm
        spine_sum = [0.0,0.0,0.0]

        #substract CENTROID and normalize each coordenate
        for i,ske_frame in enumerate(trial_queue[p]):
            trial_queue[p][i].spine = (trial_queue[p][i].spine - centroid)/normalizer
            trial_queue[p][i].hand_left = (trial_queue[p][i].hand_left - centroid)/normalizer
            trial_queue[p][i].hand_right = (trial_queue[p][i].hand_right - centroid)/normalizer
            trial_queue[p][i].elbow_left = (trial_queue[p][i].elbow_left - centroid)/normalizer
            trial_queue[p][i].elbow_right = (trial_queue[p][i].elbow_right - centroid)/normalizer
            trial_queue[p][i].neck = (trial_queue[p][i].neck - centroid)/normalizer

    return trial_queue

def extract_attributes(trial_queue, n=6):
    """extracting attributes from data"""

    attributes_queue = []
    label_list = []

    for p,essay_aux in enumerate(trial_queue):
        samples = len(essay_aux)
        div = int(samples/n)
        select = 0
        buffer_queue = []
        buffer_queue2 = []
        for x in xrange(0,n):
            buffer_queue.append(essay_aux[select])
            select += div
        for x in xrange(0,n-1):
            dif_hand_left = buffer_queue[x+1].hand_left - buffer_queue[x].hand_left
            dif_hand_right = buffer_queue[x+1].hand_right - buffer_queue[x].hand_right
            dif_elbow_left = buffer_queue[x+1].elbow_left - buffer_queue[x].elbow_left
            dif_elbow_right = buffer_queue[x+1].elbow_right - buffer_queue[x].elbow_right
            attributes_frame = np.array([dif_hand_left,dif_hand_right,dif_elbow_left,dif_elbow_right])
            buffer_queue2.append(attributes_frame)
        attributes_queue.append(buffer_queue2)
        label_list.append(essay_aux[-1].label)

    return attributes_queue, label_list


data_set = load_skeleton("skeletonData.txt")
processed_data = center_normalize_coordenates(data_set)
attributes, labels = extract_attributes(processed_data)
test = attributes.pop()
label_test = labels.pop()
data_test = np.array(test)
data_test = data_test.reshape(1,60)

#Converting attributes to numpy array to reshape
attributes_input = np.array(attributes)
labels_input = np.array(labels)

#Reshape from 2samples * 5Frames * 4Points * 3coordenates to 2samples*60attributes
attributes_input_new = attributes_input.reshape(len(labels_input),60)
print len(attributes_input_new)

#Bayesian Classifier
clf = GaussianNB()
clf.fit(attributes_input_new,labels_input)

print "The prediction is:"

print(clf.predict(data_test))
#print(clf.predict(attributes_input_new[-1]))
print "The real value is {}".format(label_test)

print "DONE"
