import torch
import torch.optim as optim
from tmp import *
from funcsVar import EncoderRNNNo, EncoderRNNMo, DecoderRNNNo, DecoderRNNMo, trainNo, trainMo, testNo, testMo


random.seed(420)
np.random.seed(420)
torch.manual_seed(420)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("Device in use:", device)

thresser = 5
NUM_INPUTS = 27 #No. of possible characters
NUM_OUTPUTS = 11  # (0-9 + '#')

### Hyperparameters and general configs
SEQ_LEN = 150
MIN_SEQ_LEN = 50

VOCAB = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
ALPHA = 0.3
MAX_LEN = 10

# Hidden size of enc and dec need to be equal if last hidden of encoder becomes init hidden of decoder
# Otherwise we would need e.g. a linear layer to map to a space with the correct dimension
NUM_UNITS_ENC = NUM_UNITS_DEC = 64
TEST_SIZE = 240
EPOCHS = 25


#assert TRAINING_SIZE % BATCH_SIZE == 0
#LEARNING_RATE=0.001
MAX_SEQ_LEN = 8
#MIN_SEQ_LEN = 5
BATCH_SIZE = 8
TRAINING_SIZE = SEQ_NUM
LEARNING_RATE = 0.005


numPrintTing = 10



encoder = EncoderRNN(NUM_INPUTS, NUM_UNITS_ENC, SEQ_LEN).to(device)
decoder = DecoderRNN(NUM_UNITS_DEC, NUM_OUTPUTS, MAX_LEN).to(device)

enc_optimizer = optim.RMSprop(encoder.parameters(), lr=LEARNING_RATE,weight_decay=1e-6,momentum=0.5)
dec_optimizer = optim.RMSprop(decoder.parameters(), lr=LEARNING_RATE,weight_decay=1e-6,momentum=0.5)

encoderNo = EncoderRNNNo(NUM_INPUTS, NUM_UNITS_ENC).to(device)
decoderNo = DecoderRNNNo(NUM_UNITS_DEC, NUM_OUTPUTS).to(device)

encoderMo = EncoderRNNMo(NUM_INPUTS, NUM_UNITS_ENC).to(device)
decoderMo = DecoderRNNMo(NUM_UNITS_DEC, NUM_OUTPUTS).to(device)



#enc_optimizer = optim.Adagrad(encoder.parameters(), lr=LEARNING_RATE, weight_decay=0e-6, initial_accumulator_value=0.1)
#dec_optimizer = optim.Adagrad(decoder.parameters(), lr=LEARNING_RATE, weight_decay=0e-6, initial_accumulator_value=0.1)



enc_optimizerNo = optim.RMSprop(encoderNo.parameters(), lr=LEARNING_RATE,weight_decay=1e-6,momentum=0.5)
dec_optimizerNo = optim.RMSprop(decoderNo.parameters(), lr=LEARNING_RATE,weight_decay=1e-6,momentum=0.5)

enc_optimizerMo = optim.RMSprop(encoderMo.parameters(), lr=LEARNING_RATE,weight_decay=1e-6,momentum=0.5)
dec_optimizerMo = optim.RMSprop(decoderMo.parameters(), lr=LEARNING_RATE,weight_decay=1e-6,momentum=0.5)




criterion = nn.NLLLoss(ignore_index=0)#,reduction='sum')
#print(encoder.parameters())
# Get training set
_,_,t1,t2,t3,lenEnc,lenDec = generateVarLen(SEQ_NUM, VOCAB, ALPHA, MAX_LEN, SEQ_LEN, MIN_SEQ_LEN)
#_,_,t1,t2,t3,lenEnc,lenDec = generateMixed(SEQ_NUM, VOCAB, ALPHA,0.5,thresser, MAX_LEN, SEQ_LEN, MIN_SEQ_LEN)

#print(t1.size(),t2.size(),t3.size())
#inputs, _, targets_in, targets, targets_seqlen, _, _, _, text_targ = generate(TRAINING_SIZE, min_len=MIN_SEQ_LEN, max_len=MAX_SEQ_LEN)
#max_target_len = max(targets_seqlen)
#inputs = torch.tensor(inputs).long()
#targets = torch.tensor(targets).long()
#targets_in = torch.tensor(targets_in).long()
#unique_text_targets = set(text_targ)

# Get validation set    
Fullval,Shortval,t1val,t2val,t3val,lenEncVal,lenDecVal = generateVarLen(SEQ_NUM, VOCAB, ALPHA, MAX_LEN, SEQ_LEN, MIN_SEQ_LEN)
#Fullval,Shortval,t1val,t2val,t3val,lenEncVal,lenDecVal = generateMixed(SEQ_NUM, VOCAB, ALPHA,0.5,thresser,MAX_LEN, SEQ_LEN, MIN_SEQ_LEN)

#val_inputs = torch.tensor(val_inputs)
#val_targets = torch.tensor(val_targets)
#val_targets_in = torch.tensor(val_targets_in)
#max_val_target_len = max(val_targets_seqlen)
#test(encoder, decoder, val_inputs, val_targets, val_targets_in, criterion, max_val_target_len)

#print('DATA: ',t1val,'\n',t2val,'\n',t3val,'\n',Fullval,'\n',Shortval)

#print(t1val[0],t2val[0],t3val[0],Fullval[0],Shortval[0])
#print('hejejej')
#print(skrald)

t1=t1.type(torch.LongTensor)
t2=t2.type(torch.LongTensor)
t3=t3.type(torch.LongTensor)

t1val=t1val.type(torch.LongTensor)
t2val=t2val.type(torch.LongTensor)
t3val=t3val.type(torch.LongTensor)

# Split training set in batches
inputs = [t1[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(TRAINING_SIZE // BATCH_SIZE)]
targets = [t2[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(TRAINING_SIZE // BATCH_SIZE)]
targets_in = [t3[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(TRAINING_SIZE // BATCH_SIZE)]
lenEnc = [lenEnc[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(TRAINING_SIZE // BATCH_SIZE)]
lenDec = [lenDec[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(TRAINING_SIZE // BATCH_SIZE)]

#inputs, targets, targets_in, lenDec, lenEnc = batchSorter(inputs, targets, targets_in, lenDec, lenEnc)

#[t1val, t2val, t3val, lenDecVal, lenEncVal, _] = Sorter([t1val, t2val, t3val, lenDecVal, lenEncVal],lenEncVal)


# Quick and dirty - just loop over training set without reshuffling

#print('MAIN: targets = ',inputs[0],'MAIN: lengths = ', lengths[0])


for epoch in range(1, EPOCHS + 1):
    train(encoder, decoder, inputs, targets, targets_in, criterion, enc_optimizer, dec_optimizer, epoch, MAX_LEN, lenEnc, lenDec)
    _, loss, accuracy = test(encoder, decoder, t1val, t2val, t3val, criterion, MAX_LEN, lenEncVal, lenDecVal)
    print('\nTest set: Average loss: {:.4f} \tAccuracy: {:.3f}%\n'.format(loss, accuracy.item()*100.))

    # Show examples
    print("Examples: prediction | input")
    out, _, _ = test(encoder, decoder, t1val[:numPrintTing], t2val[:numPrintTing], t3val[:numPrintTing], criterion, MAX_LEN, lenEncVal[:numPrintTing], lenDecVal[:numPrintTing])
    pred = get_pred(out)
    pred_text = [numbers_to_text(sample) for sample in pred]
    for i in range(numPrintTing):
        print(pred_text[i], "\t", Shortval[i], "\t" ,t2val[i])
    #print()

#print(Fullval,Shortval,t1val,t2val)    