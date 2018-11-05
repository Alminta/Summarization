import numpy as np
import random

def generate(seqNum,seqLen,vocab,alpha):
    
    length=len(vocab)
    
    listFull = []
    listShort = []
    
    for i in range(seqNum):
        listFull.append('')
        listShort.append('')
        for j in range(seqLen):
            if random.uniform(0,1)<alpha:
                date = str(np.random.randint(0,9))
                listFull[i] += date
                listShort[i] += date
            else:
                word = vocab[np.random.randint(0,length)]
                listFull[i] += word               
        
        
    
    return listFull, listShort


