import argparse
import torch
from utils_text import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
import numpy as np

parser = argparse.ArgumentParser(description='Parameter Processing')
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

embedding_size, max_sentence_len, num_classes, class_names, dst_train, dst_test, testloader = get_dataset("dbpedia", "")

# net = get_network("LSTMNet", embedding_size, num_classes).to(device)
net = get_network("MLP", embedding_size, num_classes).to(device)
optim = torch.optim.Adam(net.parameters(), lr=0.001)#, weight_decay=0.05)
criterion = torch.nn.CrossEntropyLoss().to(device)

print(np.unique(dst_train.labels))
trainloader = torch.utils.data.DataLoader(dst_train, batch_size=32, shuffle=True, num_workers=0)
for epoch in range(100):
    loss_avg = 0
    acc_avg = 0
    num_exp = 0
    for batch in trainloader:
        optim.zero_grad()

        output = net(batch[0].to(device))
        lab = batch[1].long().to(device)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
        n_b = lab.shape[0]

        loss.backward()
        optim.step()

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

    if (epoch+1)%1==0:
        print(f"Train loss: {loss_avg/num_exp}, acc: {acc_avg/num_exp}")

    if (epoch+1)%1==0:
        loss_avg = 0
        acc_avg = 0
        num_exp = 0
        for batch in testloader:
            with torch.no_grad():
                output = net(batch[0].to(device))
            lab = batch[1].long().to(device)
            loss = criterion(output, lab)
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
            n_b = lab.shape[0]

            loss_avg += loss.item()*n_b
            acc_avg += acc
            num_exp += n_b
        print(f"\nEval loss: {loss_avg/num_exp}, acc: {acc_avg/num_exp}\n")