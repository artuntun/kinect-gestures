
import nltk

f = open("skeltonData.txt","r")
print f.name
print "UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
line = "sa"

while(line !="\n"):
    line = f.readline()
    tokens = nltk.word_tokenize(f.readline())
    i = 0
    sum = 0
    for x in tokens:
        if i != 0:
            sum = sum + float(x)
        i = i + 1
    
print "UAAAAHROArrrrrrrr11111!"