
def remove_carriege(file):
    """Read Data from file and generate a list of essays"""

    with open(file, 'r') as f:
        essay_list = list(f)

    count = "0"
    with open(file, 'w') as f:
        for line in essay_list:
            if line != "\n":
                f.write(line)
            else:
                f.write("\r\n")

remove_carriege("skeletonData.txt")

print "Done"
