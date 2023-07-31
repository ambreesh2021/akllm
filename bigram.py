import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

## read input.txt
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()
# print("length of text: {} characters".format(len(text)))
# print(text[:1000])

## unique characters from text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(''.join(chars))
# print(vocab_size)

## create a mapping from unique chareacter to indices
stoi = {u:i for i, u in enumerate(chars)}
itos = {i:u for i, u in enumerate(chars)}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: ''.join([itos[c] for c in x])
# print(encode('hii there'))
# print(decode(encode('hii there')))

# encoding depends on the toekenizer used - character, sub word or word and also on relative position of word in the sentence
# import tiktoken as tk
# enc = tk.get_encoding('gpt2')
# print(enc.n_vocab)
# print(enc.encode('hii there'))
# print(enc.encode('there hii'))
# print(enc.decode(enc.encode("hii there")))

## encode the text
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:100])

## train/test
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# time dimension
# block_size = 8
# batch dimension
# batch_size = 4
# fix random seed
# torch.manual_seed(1337)

def get_batch(split):
    # generate small batch of data of input x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)
# print('----')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


#super simple bigram language model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the naxt token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensors of integers
        logits = self.token_embedding_table(idx)
        B, T, C = logits.shape

        if targets is None:
            loss = None
        else:
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is a (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # foucs on last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) #(B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append smapled index to the running sequence 
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

#logits, loss = m(xb, yb)
#print(logits.shape)
#print(loss)
#print(decode(m.generate(idx=torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

#create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#batch_size = 32
#max_iters = 1000

for iter in range(max_iters):

    # every once in a while print out the loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: loss = {losses['train']:.4f}, val_loss = {losses['val']:.4f}")
        
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))