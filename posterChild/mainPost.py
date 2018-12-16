from funcsPost import *

random.seed(420)
np.random.seed(420)
torch.manual_seed(420)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("Device in use:", device)

NUM_INPUTS = 28 #No. of possible characters
NUM_OUTPUTS = 11  # (0-9 + '#')

### Hyperparameters and general configs
SEQ_LEN = 100

VOCAB = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
ALPHA = 0.5
MAX_LEN = 10

# Hidden size of enc and dec need to be equal if last hidden of encoder becomes init hidden of decoder
# Otherwise we would need e.g. a linear layer to map to a space with the correct dimension
NUM_UNITS_ENC = NUM_UNITS_DEC = 2+SEQ_LEN
TEST_SIZE = 240
EPOCHS = 25


#assert TRAINING_SIZE % BATCH_SIZE == 0
#LEARNING_RATE=0.001
MAX_SEQ_LEN = 8
MIN_SEQ_LEN = 5
BATCH_SIZE = 8
TRAINING_SIZE = SEQ_NUM
LEARNING_RATE = 0.001



encoderNo = EncoderRNNNo(NUM_INPUTS, NUM_UNITS_ENC).to(device)
decoderNo = DecoderRNNNo(NUM_UNITS_DEC, NUM_OUTPUTS).to(device)

encoderMo = EncoderRNNMo(NUM_INPUTS, NUM_UNITS_ENC).to(device)
decoderMo = DecoderRNNMo(NUM_UNITS_DEC, NUM_OUTPUTS).to(device)

enc_optimizerNo = optim.RMSprop(encoderNo.parameters(), lr=LEARNING_RATE,weight_decay=1e-6,momentum=0.5)
dec_optimizerNo = optim.RMSprop(decoderNo.parameters(), lr=LEARNING_RATE,weight_decay=1e-6,momentum=0.5)

enc_optimizerMo = optim.RMSprop(encoderMo.parameters(), lr=LEARNING_RATE,weight_decay=1e-6,momentum=0.5)
dec_optimizerMo = optim.RMSprop(decoderMo.parameters(), lr=LEARNING_RATE,weight_decay=1e-6,momentum=0.5)
criterion = nn.NLLLoss()
#print(encoder.parameters())
# Get training set
_,_,t1,t2,t3 = generate(SEQ_NUM, SEQ_LEN, VOCAB, ALPHA,MAX_LEN)
#print(t1.size(),t2.size(),t3.size())
#inputs, _, targets_in, targets, targets_seqlen, _, _, _, text_targ = generate(TRAINING_SIZE, min_len=MIN_SEQ_LEN, max_len=MAX_SEQ_LEN)
#max_target_len = max(targets_seqlen)
#inputs = torch.tensor(inputs).long()
#targets = torch.tensor(targets).long()
#targets_in = torch.tensor(targets_in).long()
#unique_text_targets = set(text_targ)

# Get validation set
Fullval,Shortval,t1val,t2val,t3val = generate(SEQ_NUM, SEQ_LEN, VOCAB, ALPHA,MAX_LEN)

#print(Fullval)
#print(skrald)


#val_inputs = torch.tensor(val_inputs)
#val_targets = torch.tensor(val_targets)
#val_targets_in = torch.tensor(val_targets_in)
#max_val_target_len = max(val_targets_seqlen)
#test(encoder, decoder, val_inputs, val_targets, val_targets_in, criterion, max_val_target_len)

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

# Quick and dirty - just loop over training set without reshuffling



losslistNo = []
losslistMo = []

acclistNo = []
acclistMo = []

"""
for epoch in range(1, EPOCHS + 1):
    trainNo(encoderNo, decoderNo, inputs, targets, targets_in, criterion, enc_optimizerNo, dec_optimizerNo, epoch, SEQ_LEN)
    _, lossNo, accuracyNo = testNo(encoderNo, decoderNo, t1val, t2val, t3val, criterion, MAX_LEN)
    #YOLOSAPIEN
    tmp1 = lossNo.tolist()
    tmp2 = accuracyNo.tolist()

    losslistNo.append(tmp1)
    acclistNo.append(tmp2)
    print('\nTest set - no attention: Average loss: {:.4f} \tAccuracy: {:.3f}%\n'.format(lossNo, accuracyNo.item()*100.))
    

    # Show examples
    print("Examples - no attention: prediction | input")
    out, _, _ = testNo(encoderNo, decoderNo, t1val[:10], t2val[:10], t3val[:10], criterion, MAX_LEN)
    pred = get_pred(out)
    pred_text = [numbers_to_text(sample) for sample in pred]
    for i in range(10):
        print(pred_text[i], "\t", Shortval[i], "\t" ,t2val[i])
    #print()

    #print()

"""
for epoch in range(1, EPOCHS + 1):
    train(encoderMo, decoderMo, inputs, targets, targets_in, criterion, enc_optimizerMo, dec_optimizerMo, epoch, SEQ_LEN)
    _, lossMo, accuracyMo = test(encoderMo, decoderMo, t1val, t2val, t3val, criterion, MAX_LEN)

    tmp1 = lossMo.tolist()
    tmp2 = accuracyMo.tolist()

    losslistNo.append(tmp1)
    acclistNo.append(tmp2)

    print('\nTest set - mo attention: Average loss: {:.4f} \tAccuracy: {:.3f}%\n'.format(lossMo, accuracyMo.item()*100.))
    # Show more examples
    print("Examples - mo attention: prediction | input")
    out, _, _ = test(encoderMo, decoderMo, t1val[:10], t2val[:10], t3val[:10], criterion, MAX_LEN)
    pred = get_pred(out)
    pred_text = [numbers_to_text(sample) for sample in pred]
    for i in range(10):
        print(pred_text[i], "\t", Shortval[i], "\t" ,t2val[i])


print(losslistNo,'\n',acclistNo,'\n',losslistMo,'\n',acclistMo)

#print(Fullval,Shortval,t1val,t2val)