from funcs import *


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("Device in use:", device)

NUM_INPUTS = 27 #No. of possible characters
NUM_OUTPUTS = 11  # (0-9 + '#')

### Hyperparameters and general configs
SEQ_LEN = 15
SEQ_NUM = 200
VOCAB = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHA = 0.2

# Hidden size of enc and dec need to be equal if last hidden of encoder becomes init hidden of decoder
# Otherwise we would need e.g. a linear layer to map to a space with the correct dimension
NUM_UNITS_ENC = NUM_UNITS_DEC = 48
TEST_SIZE = 200
EPOCHS = 10
TEACHER_FORCING = True

assert TRAINING_SIZE % BATCH_SIZE == 0


encoder = EncoderRNN(NUM_INPUTS, NUM_UNITS_ENC).to(device)
decoder = DecoderRNN(NUM_UNITS_DEC, NUM_OUTPUTS).to(device)
enc_optimizer = optim.RMSprop(encoder.parameters(), lr=LEARNING_RATE,weight_decay=1e-6,momentum=0.5)
dec_optimizer = optim.RMSprop(decoder.parameters(), lr=LEARNING_RATE,weight_decay=1e-6,momentum=0.5)
criterion = nn.NLLLoss()

# Get training set
#inputs, _, targets_in, targets, targets_seqlen, _, _, _, text_targ = generate(TRAINING_SIZE, min_len=MIN_SEQ_LEN, max_len=MAX_SEQ_LEN)
#max_target_len = max(targets_seqlen)
#inputs = torch.tensor(inputs).long()
#targets = torch.tensor(targets).long()
#targets_in = torch.tensor(targets_in).long()
#unique_text_targets = set(text_targ)

# Get validation set
Full,Short = generate(SEQ_NUM, SEQ_LEN, VOCAB, ALPHA)
#val_inputs = torch.tensor(val_inputs)
#val_targets = torch.tensor(val_targets)
#val_targets_in = torch.tensor(val_targets_in)
#max_val_target_len = max(val_targets_seqlen)
#test(encoder, decoder, val_inputs, val_targets, val_targets_in, criterion, max_val_target_len)

# Split training set in batches
#inputs = [inputs[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(TRAINING_SIZE // BATCH_SIZE)]
#targets = [targets[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(TRAINING_SIZE // BATCH_SIZE)]
#targets_in = [targets_in[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(TRAINING_SIZE // BATCH_SIZE)]

# Quick and dirty - just loop over training set without reshuffling
"""
for epoch in range(1, EPOCHS + 1):
    train(encoder, decoder, inputs, targets, targets_in, criterion, enc_optimizer, dec_optimizer, epoch, max_target_len)
    _, loss, accuracy = test(encoder, decoder, val_inputs, val_targets, val_targets_in, criterion, max_val_target_len)
    print('\nTest set: Average loss: {:.4f} \tAccuracy: {:.3f}%\n'.format(loss, accuracy.item()*100.))

    # Show examples
    print("Examples: prediction | input")
    out, _, _ = test(encoder, decoder, val_inputs[:10], val_targets[:10], val_targets_in[:10], criterion, max_val_target_len)
    pred = get_pred(out)
    pred_text = [numbers_to_text(sample) for sample in pred]
    for i in range(10):
        print(pred_text[i], "\t", val_text_in[i])
    print()
"""