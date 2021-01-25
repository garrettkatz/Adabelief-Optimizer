"""
Based on:
https://pytorch.org/tutorials/beginner/transformer_tutorial.html#
"""
import os
import math
import torch
from torch import nn
from data import vocab, device, bptt, train_data, val_data, get_batch
from model import TransformerModel
from optimizers.AdaBelief import AdaBelief

dbg = not torch.cuda.is_available()

ntokens = len(vocab.stoi) # the size of vocabulary
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()


# do_nc = False
# clip = .5
# lr = 5.0 # learning rate
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# adabelief similar to PTB hparams
do_nc = True
clip = .5
lr = .1
optimizer = AdaBelief(model.parameters(), lr, betas=(.9, .999), weight_decay=1.2e-6, eps=1e-8)
scheduler = None
ckpt_name = "ab-clip.5" + ("-nc" if do_nc else "")

def nump(tensor, device):
    if torch.cuda.is_available(): return tensor.detach().cpu().numpy() # makes a copy
    return tensor.detach().numpy().copy()

def torc(ndarray, device):
    if torch.cuda.is_available(): return torch.tensor(ndarray).cuda()
    return torch.tensor(ndarray)

import time
def train():
    model.train() # Turn on the train mode
    newton_cap_log = []
    epoch_loss = 0.
    loss_buffer = None
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        grad = [nump(param.grad, device) for param in model.parameters()]
        old_data = [nump(param.data, device) for param in model.parameters()]

        if clip is not None: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        new_data = [nump(param.data, device) for param in model.parameters()]
        delt = [nd - od for (nd, od) in zip(new_data, old_data)]
        delt_sqnorm = sum([(d**2).sum() for d in delt])
        grad_sqnorm = sum([(g**2).sum() for g in grad])
        delt_dot_grad = sum([(d*g).sum() for (d,g) in zip(delt, grad)])

        # apply newton cap
        if do_nc and loss_buffer is not None and delt_dot_grad < 0:
            nc_ratio = - loss_buffer / delt_dot_grad
            if nc_ratio < 1:
                print("  enforcing cap: ratio = %f" % nc_ratio)
                for p, param in enumerate(model.parameters()):
                    param.data *= nc_ratio
                    param.data += torc(old_data[p] * (1 - nc_ratio), device)

        newton_cap_log.append(
            (delt_sqnorm, grad_sqnorm, delt_dot_grad, loss.item(), loss_buffer))

        # recalculate loss after step on current minibatch for independence
        with torch.no_grad():
            output = model(data, src_mask)
            loss_buffer = criterion(output.view(-1, ntokens), targets).item()

        epoch_loss += loss.item()
        total_loss += loss.item()
        log_interval = 1 # 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt,
                    (lr if scheduler is None else scheduler.get_lr()[0]),
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

        if dbg and batch == 10: break

    epoch_loss /= len(train_data) - 1
    return epoch_loss, newton_cap_log

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for batch, i in enumerate(range(0, data_source.size(0) - 1, bptt)):
            data, targets = get_batch(data_source, i)
            if data.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            if dbg and batch == 10: break
    return total_loss / (len(data_source) - 1)

best_val_loss = float("inf")
epochs = 20 # The number of epochs
best_model = None

train_losses, val_losses, nclogs = [], [], []
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_loss, newton_cap_log = train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    if scheduler is not None: scheduler.step()

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    nclogs.append(newton_cap_log)
    if not os.path.isdir('curve'): os.mkdir('curve')
    torch.save({'train_loss': train_losses, 'val_loss': val_losses, 'nc_logs': nclogs},
               os.path.join('curve', ckpt_name))

    if dbg and epoch == 2: break

