import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

    def forward(self, inputs, hidden, output_len, cn, teacher_forcing=False):
        # Input shape: [batch, output_len]
        # Hidden shape: [seq_len=1, batch_size, hidden_dim] (the last hidden state of the encoder)
        #print('c_0: {:}'.format(c_0))
        #print('teacher_forching{:}'.format(teacher_forcing))
              
        if teacher_forcing:
            dec_input = inputs
            embedded = self.embedding(dec_input)   # shape [batch, output_len, hidden_dim]
            out, (hidden, cn) = self.rnn(embedded, (hidden,cn))
            out = self.out(out)  # linear layer, out has now shape [batch, output_len, output_size]
            output = F.log_softmax(out, -1)
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
    # Run encoder and get last hidden state (and output)
    batch_size = x.size(0)
    enc_h, cn = encoder.init_hidden(batch_size)
    enc_out, enc_h, cn= encoder(x, enc_h,cn)

    dec_h = enc_h  # Init hidden state of decoder as hidden state of encoder
    dec_input = t_in
    out = decoder(dec_input, dec_h, max_t_len, cn, teacher_forcing)
    out = out.permute(0, 2, 1)
    # Shape: [batch_size x num_classes x out_sequence_len], with second dim containing log probabilities

    loss = criterion(out, t)
    pred = get_pred(log_probs=out)
    accuracy = (pred == t).type(torch.FloatTensor).mean()
    return out, loss, accuracy

def train(encoder, decoder, inputs, targets, targets_in, criterion, enc_optimizer, dec_optimizer, epoch, max_t_len):
    encoder.train()
    decoder.train()
    #batch_size = inputs.size(0)
    #enc_h = encoder.init_hidden(batch_size)

    for batch_idx, (x, t, t_in) in enumerate(zip(inputs, targets, targets_in)):
        
            
        out, loss, accuracy = forward_pass(encoder, decoder, x, t, t_in, criterion, max_t_len, True)
            
        
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        loss.backward()
        enc_optimizer.step()
        dec_optimizer.step()
        # INSERT YOUR CODE HERE
        
        
        if batch_idx % 200 == 0:
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