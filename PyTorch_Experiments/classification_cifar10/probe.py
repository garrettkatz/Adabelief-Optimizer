import os
import numpy as np
import torch
import main

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_probe_path(ckpt_name, epoch):
    return os.path.join("probepoint", "%s-epoch%03d" % (ckpt_name, epoch))

def save(ckpt_name, epoch, net, optimizer):
    path = get_probe_path(ckpt_name, epoch)
    torch.save({
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        }, path)

def load(ckpt_name, epoch):

    ckpt_name = "resnet-sgd-lr0.1-momentum0.9-wdecay0.0005-run0-resetFalse"
    path = get_probe_path(ckpt_name, epoch)
    probe = torch.load(path, map_location=torch.device(device))
    
    class Args(object):
        def __init__(self):
            self.batchsize = 128
            self.model = ckpt_name[:ckpt_name.index("-")]
    args = Args()

    train_loader, test_loader = main.build_dataset(args)

    try:
        net = main.build_model(args, device, ckpt=probe)
    except:
        # patch for different pytorch versions (leading "module." in state dict keys)
        for key in list(probe['net'].keys()):
            if len(key) >= len("module.") and key[:len("module.")] == "module.":
                val = probe['net'].pop(key)
                probe['net'][key[len("module."):]] = val
        net = main.build_model(args, device, ckpt=probe)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1) # lr placeholder, overwritten by state dict
    optimizer.load_state_dict(probe['optimizer'])
    
    return train_loader, test_loader, net, criterion, optimizer

if __name__ == "__main__":

    ckpt_name = "resnet-sgd-lr0.1-momentum0.9-wdecay0.0005-run0-resetFalse"
    train_loader, test_loader, net, criterion, optimizer = load(ckpt_name, epoch=100)
    net.train()

    print(optimizer)
    
    parms, grads = [], []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        print("batch %d: loss %f" % (batch_idx, loss))

        if batch_idx in [0,1]:
            grads.append([main.nump(param.grad, device) for param in net.parameters()])
            parms.append([main.nump(param.data, device) for param in net.parameters()])
        optimizer.step()
        
        if batch_idx == 1: break

    gtg = 0
    gn = [0, 0]
    pdiff = 0.
    for g in range(len(grads[0])):
        gtg += (grads[0][g]*grads[1][g]).sum()
        gn[0] += (grads[0][g]**2).sum()
        gn[1] += (grads[1][g]**2).sum()
        pdiff = max(pdiff, np.fabs(parms[1][g] - parms[0][g]).max())
    gn = [gn[0]**.5, gn[1]**.5]
    print("max param diff = %f" % pdiff)
    print("grad dot = %f" % gtg)
    print("grad norm = %f" % gn[0])
    print("grad norm = %f" % gn[1])
    print("grad cos = %f" % (gtg / (gn[0]*gn[1])))
    
