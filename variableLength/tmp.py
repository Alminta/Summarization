import numpy as np
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimf
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

device = "cpu"
SEQ_NUM = 800
TRAINING_SIZE = SEQ_NUM
TEACHER_FORCING = True




def generateMixed(seqNum,vocab,alpha,alphaGenNum,thres,maxLen,seqLen,minSeqLen):
    """
    Documentation here. (;
    """
    
    target = ['zero','one','two','three','four','five','six','seven','eight','nine']
    
    #target = vocabout
    #EOS = '#'
    EOSNum = len(vocab)

    length=len(vocab)
    
    t1 = torch.zeros(seqNum,seqLen).type(torch.IntTensor)
    t2 = torch.zeros(seqNum,maxLen).type(torch.IntTensor)
    t3 = torch.zeros(seqNum,maxLen).type(torch.IntTensor)
    lenEnc = torch.zeros(seqNum,1).type(torch.IntTensor)
    lenDec = torch.zeros(seqNum,1).type(torch.IntTensor)

    listFull = []
    listShort = []
    for i in range(seqNum):
        listFull.append('')
        listShort.append('')
        i1=0
        i2=0
        numNums = 0
        rndLen = np.random.randint(minSeqLen,seqLen)
 
        while (len(listFull[i])+5)<(rndLen):
            if (random.uniform(0,1)<alpha and numNums<maxLen-1) or (i1 > rndLen-12 and i2 == 0): # Generate a number
                
                date = np.random.randint(0,9)
                
                listShort[i] += str(date)
                numNums += 1

                if date>=thres or random.uniform(0,1)<alphaGenNum: #Generate a number instead of number-word
                    listFull[i] += str(date) + ' '
                    t1[i,i1] = len(vocab) + date
                    i1 += 1

                else: #Generate a number-word
                    listFull[i] += target[date] + ' '
                    for w in target[date]:
                        index = vocab.index(w)
                        t1[i,i1] = index
                        i1 += 1
                        
                #Add a space to the sequence
                t1[i,i1] = len(vocab)-1
                i1 += 1
                
                #Add the number to t2
                t2[i,i2] = date
                i2 += 1   

            else: # Generate a scrambled word
                word = genWord(1,5,vocab)
                while word in target: # Generate new words if they appear as a number-word
                    word = genWord(1,5,vocab)
                listFull[i] += word + ' '
                listShort[i] += 'w'
                
                for w in word:
                    index = vocab.index(w)
                    t1[i,i1] = index

                    i1 += 1

                t1[i,i1] = len(vocab)-1
                i1 += 1
        listFull[i]=listFull[i][:-1]

        t3[i,1:i2+1]=t2[i,:i2]
        t2[i,i2]=10 #EOSNum
        t3[i,0]=10 #EOSNUM

        lenEnc[i]=i1
        lenDec[i]=i2
        
            
    return listFull, listShort, t1, t2, t3, lenEnc, lenDec

def generateVarLen(seqNum,vocab,alpha,maxLen,seqLen,minSeqLen):
    """
    Documentation here.
    """
    
    target = ['zero','one','two','three','four','five','six','seven','eight','nine']
    #EOS = '#'
    EOSNum = len(vocab)

    length=len(vocab)
    
    t1 = torch.zeros(seqNum,seqLen).type(torch.IntTensor)
    t2 = torch.zeros(seqNum,maxLen).type(torch.IntTensor)
    t3 = torch.zeros(seqNum,maxLen).type(torch.IntTensor)
    lenEnc = torch.zeros(seqNum,1).type(torch.IntTensor)
    lenDec = torch.zeros(seqNum,1).type(torch.IntTensor)

    listFull = []
    listShort = []
    for i in range(seqNum):
        listFull.append('')
        listShort.append('')
        i1=0
        i2=0
        numNums = 0
        rndLen = np.random.randint(minSeqLen,seqLen)
 
        while (len(listFull[i])+5)<(rndLen):
            if i1 == 0:
                date = np.random.randint(1,9)
                listFull[i] += target[date] + ' '
                listShort[i] += str(date)
                numNums += 1

                for w in target[date]:
                    index = vocab.index(w)
                    t1[i,i1] = index
                    i1 += 1
                t1[i,i1] = len(vocab)-1
                i1 += 1
                
                t2[i,i2] = date
                i2 += 1   
            
            elif random.uniform(0,1)<alpha and numNums<maxLen-1:
                date = np.random.randint(1,9)
                listFull[i] += target[date] + ' '
                listShort[i] += str(date)
                numNums += 1

                for w in target[date]:
                    index = vocab.index(w)
                    t1[i,i1] = index
                    i1 += 1
                t1[i,i1] = len(vocab)-1
                i1 += 1
                t2[i,i2] = date
                i2 += 1    
            else:
                word = genWord(3,5,vocab)
                while word in target:
                    word = genWord(3,5,vocab)                
                listFull[i] += word + ' '
                listShort[i] += 'w'
                
                for w in word:
                    index = vocab.index(w)
                    t1[i,i1] = index

                    i1 += 1

                t1[i,i1] = len(vocab)-1
                i1 += 1
        listFull[i]=listFull[i][:-1]

        t3[i,1:i2+1]=t2[i,:i2]
        t2[i,i2]=10 #EOSNum
        t3[i,0]=10 #EOSNUM

        lenEnc[i]=i1
        lenDec[i]=i2
        
            
    return listFull, listShort, t1, t2, t3, lenEnc, lenDec

def generateSlightlyOld(seqNum,seqLen,vocab,alpha,maxLen):
    target = ['zero','one','two','three','four','five','six','seven','eight','nine']
    #EOS = '#'
    EOSNum = len(vocab)

    length=len(vocab)
    
    t1 = torch.zeros(seqNum,seqLen).type(torch.IntTensor)
    t2 = torch.zeros(seqNum,maxLen).type(torch.IntTensor)
    t3 = torch.zeros(seqNum,maxLen).type(torch.IntTensor)
    
    listFull = []
    listShort = []
    for i in range(seqNum):
        listFull.append('')
        listShort.append('')
        i1=0
        i2=0
        numNums = 0
        while (len(listFull[i])+5)<(seqLen):

            if random.uniform(0,1)<alpha and numNums<maxLen-1:
                date = np.random.randint(0,9)
                listFull[i] += target[date] + ' '
                listShort[i] += str(date)
                numNums += 1

                for w in target[date]:
                    index = vocab.index(w)
                    t1[i,i1] = index
                    i1 += 1
                t1[i,i1] = len(vocab)-1
                i1 += 1
                t2[i,i2] = date
                i2 += 1    
            else:
                word = genWord(3,5,vocab)
                while word in target:
                    word = genWord(3,5,vocab)                
                listFull[i] += word + ' '
                listShort[i] += 'w'
                
                for w in word:
                    index = vocab.index(w)
                    t1[i,i1] = index
                    #t2[i,i2] = index
                    i1 += 1
                    #i2 += 1
                t1[i,i1] = len(vocab)-1
                i1 += 1
        listFull[i]=listFull[i][:-1]
        #listFull[i] += EOS
        #listShort[i] += EOS
        t3[i,1:i2+1]=t2[i,:i2]
        t2[i,i2]=10#EOSNum
        t3[i,0]=10#EOSNUM
        #t2[i,i2]=EOSNum
      
        
    #t2,_=torch.sort(t2,1)
    
    return listFull, listShort, t1, t2, t3

def genWord(minLen,maxLen,vocab):
    r = random.randint(minLen,maxLen)
    word = ''

    for i in range(r):
        word += vocab[random.randint(0,len(vocab)-2)]

    return word

def generateOld(seqNum,seqLen,vocab,alpha,maxLen):
    target = ['zero','one','two','three','four','five','six','seven','eight','nine']


    length=len(vocab)
    
    t1 = torch.zeros(seqNum,seqLen).type(torch.IntTensor)
    t2 = torch.zeros(seqNum,maxLen).type(torch.IntTensor)

    listFull = []
    listShort = []
    
    for i in range(seqNum):
        listFull.append('')
        listShort.append('')
        k=0
        for j in range(seqLen):
            if random.uniform(0,1)<alpha and len(listShort[i])<maxLen:
                date = np.random.randint(1,9)
                listFull[i] += str(date)
                listShort[i] += str(date)
                t1[i,j] = date
                t2[i,k] = date
                k += 1
            else:
                word = np.random.randint(10,10+length)
                listFull[i] += str(vocab[word-10])
                t1[i,j] = word

        
        
    t2,_=torch.sort(t2,1)
    
    return listFull, listShort.sort(), t1, t2



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, maksimus):
        super().__init__()
        self.hidden_size = hidden_size
        self.maksimus = maksimus

        self.embedding = nn.Embedding(input_size, self.hidden_size)
        rnn = nn.LSTM
        self.rnn = rnn(self.hidden_size, self.hidden_size, 1, batch_first=True)

    def forward(self, inputs, hidden, cn, lenenc):
        # Input shape [batch, seq_in_len]z
        inputs = inputs.long()
        #print(inputs.shape)
        #print(inputs)
        # Embedded shape [batch, seq_in_len, embed]
        [inputs,lenenc,index] = Sorter([inputs,lenenc],lenenc)
        embedded = self.embedding(inputs.long())
        #print('ENCODER: lengths',lengths.shape,lengths)
        
        embedded = pack_padded_sequence(embedded,lenenc.squeeze(1).tolist(),batch_first=True)
        # Output shape [batch, seq_in_len, embed]
        # Hidden shape [1, batch, embed], last hidden state of the GRU cell
        # We will feed this last hidden state into the decoder
        output, (hidden,cn) = self.rnn(embedded, (hidden,cn))
        (output,_) = pad_packed_sequence(output,batch_first=True,total_length=self.maksimus)
        #print('ENCODER: output = ',output.shape,'\n ENCODER: hidden = ',hidden.shape)
        output = unSorter(output,index)
        hidden = unSorter(hidden.squeeze(0),index)
        cn = unSorter(cn.squeeze(0),index)
        return output, hidden.unsqueeze(0), cn.unsqueeze(0)

    def init_hidden(self, batch_size):
        init = torch.zeros(1, batch_size, self.hidden_size, device=device)
        cn = torch.zeros(1,batch_size, self.hidden_size, device=device)
        return init, cn


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, maks_len):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.maks_len = maks_len

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        rnn = nn.LSTM
        self.rnn = rnn(self.hidden_size, self.hidden_size, 1, batch_first=True)
        

        # Attention

        #.W_2 = Parameter(init.kaiming_normal_(torch.Tensor(num_output, num_hidden))) #template

        self.v_a = Parameter(nn.init.kaiming_normal_(torch.Tensor(hidden_size,1)))

        self.W_h = Parameter(nn.init.kaiming_normal_(torch.Tensor(hidden_size, hidden_size)))

        self.W_s = Parameter(nn.init.kaiming_normal_(torch.Tensor(hidden_size, hidden_size)))

        #self.w_c = Parameter(init.kaiming_normal_(torch.Tensor(num_output, num_hidden))) need to be defined

        self.b_alpha = Parameter(nn.init.constant_(torch.Tensor(hidden_size),0))

        self.V_in = nn.Linear(self.hidden_size*2,self.hidden_size,bias=True)

        self.V_out = nn.Linear(self.hidden_size,self.output_size,bias=True)
        """
        self.VV_1 = Parameter(nn.init.kaiming_normal_(torch.Tensor(hidden_size*2,output_size)))

        self.VV_2 = Parameter(nn.init.kaiming_normal_(torch.Tensor(output_size,output_size)))

        self.bb_1 = Parameter(nn.inin.constant_(torch.Tensor(output_size),0))

        self.bb_2 = Parameter(nn.inin.constant_(torch.Tensor(output_size),0))
        """



    def forward(self, inputs, hidden, output_len, cn, encoder_out, lendec, teacher_forcing=False):
        # Input shape: [batch, output_len]
        # Hidden shape: [seq_len=1, batch_size, hidden_dim] (the last hidden state of the encoder)
        #print('c_0: {:}'.format(c_0))
        #print('teacher_forching{:}'.format(teacher_forcing))
              
        if teacher_forcing:
            dec_input = inputs
            denNyeKonge = torch.zeros(hidden.shape[1],self.output_size,dec_input.shape[1]) # Init the new king. Cheers!
            #print('FORWARD: dec_input = ', dec_input)
            [dec_input,lendec,index] = Sorter([dec_input,lendec],lendec)
            #print('FORWARD: dec_input_sort = ', dec_input)


            
            #print('FORWARD: test = ',test)

            #print('FORWARD: outputsize = ',self.output_size,'\nFORWARD: hiddensize = ',self.hidden_size,'\nFORWARD: decinput = ',dec_input.shape,'\n',dec_input)
            embedded = self.embedding(dec_input.type(torch.LongTensor))
            #print(dec_input)
            #print('FORWARD: lendec = ',lendec)
            #print('FORWARD: embedded = ', embedded)
            #print('FORWARD: embedded.shape= ',embedded.shape)
            #[embedded,index_rev,index] = Sorter([embedded],lendec)
            #print('FORWARD: index = ',index)
            #print('FORWARD: index_rev = ',index_rev)
            #print('FORWARD: embedded_sort = ',embedded)
            #print('FORWARD: embedded_sort.shape= ',embedded.shape)

            embedded = pack_padded_sequence(embedded,lendec.squeeze(1).tolist(),batch_first=True)
            out, (hidden, cn) = self.rnn(embedded, (hidden,cn))
            (out,_) = pad_packed_sequence(out,batch_first=True,total_length=self.maks_len)
            
            out = unSorter(out,index)
            #print('FORWARD: out = ',out,'and the shape is = ',out.shape)
            
            
            #print('FORWARD: hidden =',hidden.shape,'\n FORWARD: dennyekonge = ', denNyeKonge.shape,'\n FORWARD: embedded = ',embedded.shape,'\n FORWARD: dec_input = ',dec_input.shape)
            #out, (hidden, cn) = self.rnn(embedded, (hidden,cn))
            #print('Skaal!')
            #print(nyeK)

            #print('FORWARD: encoder_out = ',encoder_out[0].shape)
            part1 = torch.matmul(encoder_out,self.W_h) # gives the part1 dimension [ B, T, W_h[1] ] since W_h converts from [ d ] to [ W_h[1] ]
            for i in range(dec_input.shape[1]): # Does some attention (:
                #print('FORWARD: EMBEDDED = ',embedded.shape,'FORWARD: hidden = ',hidden.shape)
                #print('FORWARD: lennn = ',lengths.squeeze(1).tolist())
                #print('FORWARD: embers = ',embedded[:,i,...].shape,'\nFORWARD: embers raw = ',embedded[:,i,...])
                #embeddedi = pack_padded_sequence(embedded,lengths.squeeze(1).tolist(),batch_first=True)
                #print('FORWARD: EMB I = ',embeddedi)
                #out, (hidden, cn) = self.rnn(embeddedi, (hidden,cn))
                #out = pad_packed_sequence(out,batch_first=True)
                #print('\n FORWARD: out = ',out.shape,'\n FORWARD: encoder_out = ',encoder_out.shape)
                #print('FOWRAD: out i  = ',out[:,i,...].shape)
                part2 = torch.matmul(out[:,i,...].unsqueeze(1), self.W_s) # gives the part2 dimension [ B, T, W_s[1] ] since W_s converts from [ d ] to [ W_s[1] ]
                #print('FORWARD: part1 = ',part1.shape,'\n FORWARD: part2 = ',part2.shape,'\n FORWARD: biazz = ',self.b_alpha.unsqueeze(0).unsqueeze(0).shape)
                bjorn = part1 + part2 + self.b_alpha.unsqueeze(0).unsqueeze(0)
                #print('FORWARD: part1 + part2', (part1 + part2.unsqueeze(1)).shape)
                #print('FORWARD: Bjorn = ',bjorn.shape, '\n FORWARD: tanh(bjorn) = ', F.tanh(bjorn).shape, '\n FORWARD: v_a = ', self.v_a.shape)
                #print('FORWARD: tanh(bjorn).type = ', F.tanh(bjorn).type)
                #print('FORWARD: v_a * tanh(bjorn) = ', torch.matmul(F.tanh(bjorn),self.v_a).shape)
                
                Cool = F.softmax(torch.matmul(torch.tanh(bjorn),self.v_a),dim=1)
                #print('FORWARD: cool = ',Cool.shape,'\n FORWARD: cool sq = ',Cool.squeeze(2).shape)
                #
                # IMPLEMENT POINTER HERE.
                #
                #

                #print(Cool)
                #print(Kewl)
                #hStar = torch.matmul(encoder_out,Cool)
                hStar = torch.sum(encoder_out * Cool,dim=1)
                #print('FORWARD: hStar2 = ',hStar.shape)
                #print(kska)
                #Cool = torch.cat((out,Cool))
                #print('FORWARD: out = ',out.shape)
                #print('ud = ',out[0,-1])
                #print(nejtak)
                tmp = F.log_softmax(self.V_out(self.V_in(torch.cat((out[:,i,...],hStar),dim=1))),dim=1)
                #print('FORWARD: tmp = ',tmp.shape,'\n FORWARD: tmp == ',tmp)

                #tmp2 = F.log_softmax(torch.cat((out.squeeze(1),hStar),dim=1))

                denNyeKonge[...,i] = tmp


            output = denNyeKonge
            #out = self.out(denNyeKonge)  # linear layer, out has now shape [batch, output_len, output_size]
            #output = F.log_softmax(alpha, -1)
        else:
            # Take the EOS character only, for the whole batch, and unsqueeze so shape is [batch, 1]
            # This is the first input, then we will use as input the GRU output at the previous time step
            dec_input = inputs[:, 0].unsqueeze(1)

            output = []
            for i in range(output_len):
                out, (hidden, cn) = self.rnn(self.embedding(dec_input), (hidden,cn))
                out = self.out(out)  # linear layer, out has now shape [batch, 1, output_size]
                out = F.log_softmax(out, -1)
                output.append(out.squeeze(1))
                out_symbol = torch.argmax(out, dim=2)   # shape [batch, 1]
                dec_input = out_symbol   # feed the decoded symbol back into the recurrent unit at next step

            output = torch.stack(output).permute(1, 0, 2)  # [batch_size x seq_len x output_size]

        return output

"""
# attention
# ASSUMING THAT enc_out HAS THE DIMENSIONS [ B, T, d ]
part1 = torch.matmul(enc_out,W_h) # gives the part1 dimension [ B, T, W_h[1] ] since W_h converts from [ d ] to [ W_h[1] ]
part1.unsqueeze(1)

# ASSUMIMG THAT out HAS THE DIMENSIONS [ B, T, d ]
part2 = torch.matmul(out, W_s) # gives the part2 dimension [ B, T, W_s[1] ] since W_s converts from [ d ] to [ W_s[1] ]
part2 = part2.unsqueeze(1)

# ASSUMING DIMENSIONS [ B, T, b ].
john = part1 + part2 + b_alpha # Summs the "boxes" part1 & part2 and adds b_alpha along the [ b ] dimension

e = np.sum(v_a * torch.tanh(john), axis=2)

alpha = F.softmax(e)

# Comment bc mask
#alpha = mask*(attention_lengths.type(torch.FloatTensor)) * alpha

# Comment bc mask
#alpha = alpha / np.sum(alpha, axis=1)

# WE DO NOT USE THE ATTENTION TRACKER I GUESS https://i.kym-cdn.com/photos/images/original/001/231/999/ba5.jpg
# Comment bc mask
#c = np.sum(alpha.unsqueeze(2)*hidden.squeeze(),axis=1)
"""



def forward_pass(encoder, decoder, x, t, t_in, criterion, max_t_len, lenenc, lendec, teacher_forcing=True):
    """
    Executes a forward pass through the whole model.

    :param encoder:
    :param decoder:
    :param x: input to the encoder, shape [batch, seq_in_len]
    :param t: target output predictions for decoder, shape [batch, seq_t_len]
    :param criterion: loss function
    :param max_t_len: maximum target length

    :return: output (after log-softmax), loss, accuracy (per-symbol)
    """
    #print('FORWARD: t_in = ',t_in.shape,'\n FORWARD: x = ',x.shape)
    #print(x.size(),'Et eller andet lort')
    # Run encoder and get last hidden state (and output)
    batch_size = x.size(0)
    enc_h, cn = encoder.init_hidden(batch_size)
    enc_out, enc_h, cn= encoder(x, enc_h, cn, lenenc)
    #print('\n')
    #print(enc_out.size())
    #print(enc_h.size())
    dec_h = enc_h  # Init hidden state of decoder as hidden state of encoder
    dec_input = t_in
    out = decoder(dec_input, dec_h, max_t_len, cn, enc_out, lendec, teacher_forcing)
    #print('FORWARD_PASS: out = ',out.shape,'FORWARD_PASS: t = ',t.shape)
    #print('OUT; ',out)
    #out = out.permute(0, 2, 1)
    # Shape: [batch_size x num_classes x out_sequence_len], with second dim containing log probabilities
    #print(hej)
    loss = criterion(out, t)
    #print('PASsAge: out = ',out,'PASSAGE: t = ',t)
    #print(ddø)
    pred = get_pred(log_probs=out)
    accuracy = (torch.max((pred == t), (t == 0))).type(torch.FloatTensor).mean()
    return out, loss, accuracy

def train(encoder, decoder, inputs, targets, targets_in, criterion, enc_optimizer, dec_optimizer, epoch, max_t_len, lenEnc, lenDec):
    encoder.train()
    decoder.train()
    #batch_size = inputs.size(0)
    #enc_h = encoder.init_hidden(batch_size)

    #print(inputs[0].size(),targets[0].size(),targets_in[0].size())
    #print(inputs)
    for batch_idx, (x, t, t_in, lenenc, lendec) in enumerate(zip(inputs, targets, targets_in, lenEnc, lenDec)):
        #print(batch_idx)
        #print(x.size()) 
        #print(lenn)
        #print(x)
        #print(crash)   
        out, loss, accuracy = forward_pass(encoder, decoder, x, t, t_in, criterion, max_t_len, lenenc, lendec, True)
            
        
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        loss.backward()
        enc_optimizer.step()
        dec_optimizer.step()
        # INSERT YOUR CODE HERE
        #print(batch_idx)
        
        if batch_idx % 50 == 0:
            encoder.eval()
            decoder.eval()
            print('Epoch {} [{}/{} ({:.0f}%)]\tTraining loss: {:.4f} \tTraining accuracy: {:.1f}%'.format(
                epoch, batch_idx * len(x), TRAINING_SIZE,
                100. * batch_idx * len(x) / TRAINING_SIZE, loss.item(),
                100. * accuracy.item()))
            encoder.train()
            decoder.train()

def batchSorter(inputs,targets,targets_in,lenDec,lenEnc):
    inputs_new = []
    targets_new = []
    targets_in_new = []
    lenDec_new = []
    lenEnc_new = []
    for i in range(len(inputs)):

        L = Sorter([inputs[i],targets[i],targets_in[i],lenDec[i],lenEnc[i]],lenEnc[i])
        inputs_new.append(L[0].type(torch.LongTensor))
        targets_new.append(L[1].type(torch.LongTensor))
        targets_in_new.append(L[2].type(torch.LongTensor))
        lenDec_new.append(L[3].type(torch.LongTensor))
        lenEnc_new.append(L[4].type(torch.LongTensor))
        '''
        inputs_next = torch.zeros(inputs[0].shape)
        targets_next = torch.zeros(targets[0].shape)
        targets_in_next = torch.zeros(targets_in[0].shape)
        lengths_next,index = torch.sort(lengths[i].squeeze(1),descending=True)     
        for j in index.tolist():
            inputs_next[j] = inputs[i][j]
            targets_next[j] = targets[i][j]
            targets_in_next[j] = targets_in[i][j]
        inputs_new.append(inputs_next.type(torch.LongTensor))
        targets_new.append(targets_next.type(torch.LongTensor))
        targets_in_new.append(targets_in_next.type(torch.LongTensor))
        lengths_new.append(lengths_next.unsqueeze(1).type(torch.LongTensor))
        '''

    return inputs_new,targets_new,targets_in_new, lenDec_new, lenEnc_new
def Sorter(List,length):
    List_new = []
    for i in List:
        List_new.append(torch.zeros(i.shape))
    length_next,index = torch.sort(length.squeeze(1),descending=True)
    index = index.tolist()
    for j in range(len(List)):
        for i in range(List[0].shape[0]):
            List_new[j][i] = List[j][index[i]]
    List_new.append(torch.IntTensor(index).unsqueeze(1))
    return List_new

def unSorter(inputt,indeks):
    output = torch.zeros(inputt.shape)
    longindeks = indeks.type(torch.LongTensor)

    for i in range(indeks.shape[0]):
        output[longindeks[i],...] = inputt[i,...]

    return output


def test(encoder, decoder, inputs, targets, targets_in, criterion, max_t_len, lenEnc, lenDec):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        targets = targets.long().to(device)
        targets_in = targets_in.long().to(device)
        out, loss, accuracy = forward_pass(encoder, decoder, inputs, targets, targets_in, criterion, max_t_len, lenEnc, lenDec, teacher_forcing=TEACHER_FORCING)
    return out, loss, accuracy


def numbers_to_text(seq):
    return "".join([str(to_np(i)) if to_np(i) != 10 else '#' for i in seq])

def to_np(x):
    return x.cpu().numpy()

def get_pred(log_probs):

    return torch.argmax(log_probs, dim=1)

# junk in the bottom
