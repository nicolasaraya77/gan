import random as rand

# ==================================
# graph.csv
# ==================================

def writeArray(val) :
    text = ''
    for i in val :
        for pos,value in enumerate(i) :
            text += str(value)
            if len(i) - 1 != pos:
                text += ','
        text += '\n'
    return text

# Create graph.csv M
f = open('data/graph.csv', 'w')

# adjacency matrix M
graph = [
[0,1,1,1,0,0,0,0,0],
[0,0,0,0,1,0,0,0,0],
[0,0,0,0,0,0,0,1,0],
[0,0,0,0,0,1,0,0,0],
[0,0,0,0,0,0,1,0,0],
[0,0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,0,0,0],
]

text = ''
for pos, _ in enumerate(graph) :
    text += 'nod' + str(pos +1)
    if pos != len(graph) - 1 :
        text += ','

text = text + '\n' + writeArray(graph)
f.write(text)
f.close()

# ==================================
# graph.csv
# ==================================



# size can be greater or equal than 9
def fillN(size) :
    n = [i+1 for i in range(9) ]
    rand.shuffle(n)
    if(size > 9) :
        for i in range(size - 9) :
            n.append(rand.randint(1,9))
    return n

# Dataset N and M
f = open('data/dset.csv', 'w')
text = ''
M = len(graph)
N = M
cases = 42

for i in range(N):
    text += 'n' + str(i+1) + ','

for i in range(M):
    text += 'm' + str(i+1)
    if M-1 != i :
        text += ','

text += '\n'
for i in range(cases):
    #text += str(i+1) + ','

    n = fillN(N)
    for pos,value in enumerate(n) :
        text += str(value) + ','

    # Como son 9 con 9 el % va a quedar como 1/9 el porcentaje y será igual en todos
    for j in range(M):
        text += str(round(rand.uniform(30.5, 100.0), 3))
        if M-1 != j :
            text += ','

    text += '\n'


f.write(text)
f.close()
