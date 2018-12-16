import numpy as np
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimf
from torch.nn.parameter import Parameter

device = "cpu"
SEQ_NUM = 800
TRAINING_SIZE = SEQ_NUM
TEACHER_FORCING = True
def generateIsh(seqNum,seqLen,vocab,alpha,maxLen,minLen):
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
    lenT = torch.zeros(seqNum,1).type(torch.IntTensor)
    listFull = []
    listShort = []
    for i in range(seqNum):
        listFull.append('')
        listShort.append('')
        i1=0
        i2=0
        numNums = 0
        while (len(listFull[i])+5)<(seqLen):
            rndNum = np.random.randint(i1,maxLen)
            #if rndNum > minLen 
            print('Everything is going wrong!!!!!!')

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




def generate(seqNum,seqLen,vocab,alpha,maxLen):
    target = ['zero','one','two','three','four','five','six','seven','eight','nine']
    #EOS = '#'
    EOSNum = len(vocab)

    length=len(vocab)
    
    t1 = torch.zeros(seqNum,seqLen).type(torch.IntTensor)
    t2 = torch.zeros(seqNum,maxLen).type(torch.IntTensor)
    t3 = torch.zeros(seqNum,maxLen).type(torch.IntTensor)
    
#*len(max(target,key=len))  
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
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, self.hidden_size)
        rnn = nn.LSTM
        self.rnn = rnn(self.hidden_size, self.hidden_size, 1, batch_first=True)

    def forward(self, inputs, hidden,cn):
        # Input shape [batch, seq_in_len]z
        inputs = inputs.long()

        # Embedded shape [batch, seq_in_len, embed]
        embedded = self.embedding(inputs)
        
        # Output shape [batch, seq_in_len, embed]
        # Hidden shape [1, batch, embed], last hidden state of the GRU cell
        # We will feed this last hidden state into the decoder
        output, (hidden,cn) = self.rnn(embedded, (hidden,cn))
        #print('ENCODER: output = ',output.shape,'\n ENCODER: hidden = ',hidden.shape)
        return output, hidden, cn

    def init_hidden(self, batch_size):
        init = torch.zeros(1, batch_size, self.hidden_size, device=device)
        cn = torch.zeros(1,batch_size, self.hidden_size, device=device)
        return init, cn


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

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



    def forward(self, inputs, hidden, output_len, cn, encoder_out, teacher_forcing=False):
        # Input shape: [batch, output_len]
        # Hidden shape: [seq_len=1, batch_size, hidden_dim] (the last hidden state of the encoder)
        #print('c_0: {:}'.format(c_0))
        #print('teacher_forching{:}'.format(teacher_forcing))
              
        if teacher_forcing:
            dec_input = inputs
            denNyeKonge = torch.zeros(hidden.shape[1],self.output_size,dec_input.shape[1]) # Init the new king. Cheers!

            #print(dec_input)
            embedded = self.embedding(dec_input)
            #print('FORWARD: hidden =',hidden.shape,'\n FORWARD: dennyekonge = ', denNyeKonge.shape,'\n FORWARD: embedded = ',embedded.shape,'\n FORWARD: dec_input = ',dec_input.shape)
            #out, (hidden, cn) = self.rnn(embedded, (hidden,cn))
            #print('Skaal!')
            #print(nyeK)
            for i in range(dec_input.shape[1]):
                out, (hidden, cn) = self.rnn(embedded[:,i,...].unsqueeze(1), (hidden,cn))
                #print('\n FORWARD: out = ',out.shape,'\n FORWARD: encoder_out = ',encoder_out.shape)

                part1 = torch.matmul(encoder_out,self.W_h) # gives the part1 dimension [ B, T, W_h[1] ] since W_h converts from [ d ] to [ W_h[1] ]
                part2 = torch.matmul(out, self.W_s) # gives the part2 dimension [ B, T, W_s[1] ] since W_s converts from [ d ] to [ W_s[1] ]
                bjorn = part1 + part2 + self.b_alpha.unsqueeze(0).unsqueeze(0)
                #print('FORWARD: part1 = ',part1.shape,'\n FORWARD: part2 = ',part2.shape,'\n FORWARD: biazz = ',self.b_alpha.unsqueeze(0).unsqueeze(0).shape)
                #print('FORWARD: part1 + part2', (part1 + part2.unsqueeze(1)).shape)
                #print('FORWARD: Bjorn = ',bjorn.shape, '\n FORWARD: tanh(bjorn) = ', F.tanh(bjorn).shape, '\n FORWARD: v_a = ', self.v_a.shape)
                #print('FORWARD: tanh(bjorn).type = ', F.tanh(bjorn).type)
                #print('FORWARD: v_a * tanh(bjorn) = ', torch.matmul(F.tanh(bjorn),self.v_a).shape)
                
                Cool = F.softmax(torch.matmul(torch.tanh(bjorn),self.v_a),dim=1)
                #print('FORWARD: cool = ',Cool.shape,'\n FORWARD: cool sq = ',Cool.squeeze(2).shape)
                
                #print(Cool)
                #print(Kewl)
                #hStar = torch.matmul(encoder_out,Cool)
                hStar = torch.sum(encoder_out * Cool,dim=1)
                #print('FORWARD: hStar2 = ',hStar.shape)
                #Cool = torch.cat((out,Cool))
                #print('FORWARD: out = ',out.shape)

                tmp = F.log_softmax(self.V_out(self.V_in(torch.cat((out.squeeze(1),hStar),dim=1))),dim=1)
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



def forward_pass(encoder, decoder, x, t, t_in, criterion, max_t_len, teacher_forcing):
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
    enc_out, enc_h, cn= encoder(x, enc_h,cn)
    #print('\n')
    #print(enc_out.size())
    #print(enc_h.size())
    dec_h = enc_h  # Init hidden state of decoder as hidden state of encoder
    dec_input = t_in
    out = decoder(dec_input, dec_h, max_t_len, cn, enc_out, teacher_forcing)
    #print('FORWARD_PASS: out = ',out.shape,'FORWARD_PASS: t = ',t.shape)
    #print('OUT; ',out)
    #out = out.permute(0, 2, 1)
    # Shape: [batch_size x num_classes x out_sequence_len], with second dim containing log probabilities
    #print(hej)
    loss = criterion(out, t)
    pred = get_pred(log_probs=out)
    accuracy = (pred == t).type(torch.FloatTensor).mean()
    return out, loss, accuracy

def train(encoder, decoder, inputs, targets, targets_in, criterion, enc_optimizer, dec_optimizer, epoch, max_t_len):
    encoder.train()
    decoder.train()
    #batch_size = inputs.size(0)
    #enc_h = encoder.init_hidden(batch_size)

    #print(inputs[0].size(),targets[0].size(),targets_in[0].size())
    #print(inputs)
    for batch_idx, (x, t, t_in) in enumerate(zip(inputs, targets, targets_in)):
        #print(batch_idx)
        #print(x.size())    
        out, loss, accuracy = forward_pass(encoder, decoder, x, t, t_in, criterion, max_t_len, True)
            
        
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


def test(encoder, decoder, inputs, targets, targets_in, criterion, max_t_len):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        targets = targets.long().to(device)
        targets_in = targets_in.long().to(device)
        out, loss, accuracy = forward_pass(encoder, decoder, inputs, targets, targets_in, criterion, max_t_len,
                                           teacher_forcing=TEACHER_FORCING)
    return out, loss, accuracy


#mitjunk

def numbers_to_text(seq):
    return "".join([str(to_np(i)) if to_np(i) != 10 else '#' for i in seq])

def to_np(x):
    return x.cpu().numpy()

def get_pred(log_probs):
    """
    Get class prediction (digit prediction) from the net's output (the log_probs)
    :param log_probs: Tensor of shape [batch_size x n_classes x sequence_len]
    :return:
    """
    return torch.argmax(log_probs, dim=1)

# junk in the bottom