from funcs import *


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("Device in use:", device)

NUM_INPUTS = 27 #No. of possible characters
NUM_OUTPUTS = 11  # (0-9 + '#')

### Hyperparameters and general configs
SEQ_LEN = 50

VOCAB = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
ALPHA = 0.2
MAX_LEN = 15

# Hidden size of enc and dec need to be equal if last hidden of encoder becomes init hidden of decoder
# Otherwise we would need e.g. a linear layer to map to a space with the correct dimension
NUM_UNITS_ENC = NUM_UNITS_DEC = 2+SEQ_LEN
TEST_SIZE = 200
EPOCHS = 10


#assert TRAINING_SIZE % BATCH_SIZE == 0
#LEARNING_RATE=0.001
MAX_SEQ_LEN = 8
MIN_SEQ_LEN = 5
BATCH_SIZE = 8
TRAINING_SIZE = SEQ_NUM
LEARNING_RATE = 0.001


encoder = EncoderRNN(NUM_INPUTS, NUM_UNITS_ENC).to(device)
decoder = DecoderRNN(NUM_UNITS_DEC, NUM_OUTPUTS).to(device)
enc_optimizer = optim.RMSprop(encoder.parameters(), lr=LEARNING_RATE,weight_decay=1e-6,momentum=0.5)
dec_optimizer = optim.RMSprop(decoder.parameters(), lr=LEARNING_RATE,weight_decay=1e-6,momentum=0.5)
criterion = nn.NLLLoss()

# Get training set
_,_,t1,t2 = generate(SEQ_NUM, SEQ_LEN, VOCAB, ALPHA,MAX_LEN)
#inputs, _, targets_in, targets, targets_seqlen, _, _, _, text_targ = generate(TRAINING_SIZE, min_len=MIN_SEQ_LEN, max_len=MAX_SEQ_LEN)
#max_target_len = max(targets_seqlen)
#inputs = torch.tensor(inputs).long()
#targets = torch.tensor(targets).long()
#targets_in = torch.tensor(targets_in).long()
#unique_text_targets = set(text_targ)

# Get validation set
Fullval,Shortval,t1val,t2val = generate(SEQ_NUM, SEQ_LEN, VOCAB, ALPHA,MAX_LEN)
#val_inputs = torch.tensor(val_inputs)
#val_targets = torch.tensor(val_targets)
#val_targets_in = torch.tensor(val_targets_in)
#max_val_target_len = max(val_targets_seqlen)
#test(encoder, decoder, val_inputs, val_targets, val_targets_in, criterion, max_val_target_len)

t1=t1.type(torch.LongTensor)
t2=t2.type(torch.LongTensor)
t1val=t1val.type(torch.LongTensor)
t2val=t2val.type(torch.LongTensor)


# Split training set in batches
inputs = [t1[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(TRAINING_SIZE // BATCH_SIZE)]
targets = [t2[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(TRAINING_SIZE // BATCH_SIZE)]
targets_in = [t2[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(TRAINING_SIZE // BATCH_SIZE)]

# Quick and dirty - just loop over training set without reshuffling

for epoch in range(1, EPOCHS + 1):
    train(encoder, decoder, inputs, targets, targets_in, criterion, enc_optimizer, dec_optimizer, epoch, SEQ_LEN)
    _, loss, accuracy = test(encoder, decoder, t1val, t2val, t2val, criterion, MAX_LEN)
    print('\nTest set: Average loss: {:.4f} \tAccuracy: {:.3f}%\n'.format(loss, accuracy.item()*100.))

    # Show examples
    print("Examples: prediction | input")
    out, _, _ = test(encoder, decoder, t1val[:10], t2val[:10], t2val[:10], criterion, MAX_LEN)
    pred = get_pred(out)
    pred_text = [numbers_to_text(sample) for sample in pred]
    for i in range(10):
        print(pred_text[i], "\t", Fullval[i])
    #print()

#print(Full,Short,t1,t2)