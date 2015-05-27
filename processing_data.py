from collections import deque, namedtuple

Coordenates = namedtuple('Coordenates', 'hand_left, elbow_left, elbow_right, hand_right, neck, spine')

class SkeletonFrame():
    """This class store data for each Skeletonframe"""

    def __init__(self, line):
        tokens = line.split()
        self.label = tokens[0]
        self.coordenates = Coordenates(tokens[1:4], tokens[4:7], tokens[7:10],
                                        tokens[10:13], tokens[13:16], tokens[16:19])

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


skeleton = trial_queue[-1].pop()

print skeleton.coordenates.spine

skeleton2 = trial_queue[-2].pop()

print skeleton2.coordenates.spine
