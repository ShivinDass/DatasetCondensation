import time
import os
import numpy as np
import torch
import pandas as pd
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
from networks import MLP, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN, LSTMNet, MLPV2
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, evaluation

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim.downloader as api
import re
import csv
import h5py

SST_DATASETS = ["SST1-w2v", "SST1-glove", "SST1-transformer", "SST1-w2v-flat", "SST1-glove-flat", "SST1-transformer-flat", \
                    "SST2-w2v", "SST2-glove", "SST2-transformer", "SST2-w2v-flat", "SST2-glove-flat", "SST2-transformer-flat"]

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

def encode_data(dataset):
    folder_path = os.path.join(os.environ['DATA_DIR'], dataset)

    ind = 2 if dataset == 'dbpedia' else 1
    x_train = []
    y_train = []
    with open(os.path.join(folder_path, 'train.csv'), newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            x_train.append(row[ind])
            y_train.append(int(row[0])-1)
            
    x_test = []
    y_test = []
    with open(os.path.join(folder_path, 'test.csv'), newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            x_test.append(row[ind])
            y_test.append(int(row[0])-1)

    # for i in range(10):
    #     print(x_train[i])
    #     print(y_train[i])
    #     print()
    # return 

    st_model = 'paraphrase-mpnet-base-v2' #@param ['paraphrase-mpnet-base-v2', 'all-mpnet-base-v1', 'all-mpnet-base-v2', 'stsb-mpnet-base-v2', 'all-MiniLM-L12-v2', 'paraphrase-albert-small-v2', 'all-roberta-large-v1']
    model = SentenceTransformer(st_model)
    embedding_size = 768

    x_eval = None
    for i in range(0, len(x_test), 1024):
        k = model.encode(x_test[i:i+1024])
        x_eval = k if x_eval is None else np.concatenate((x_eval, k), axis=0)
        print(f"test {i}/{len(x_test)}")
        
    print(x_eval.shape, len(y_test))

    with h5py.File(os.path.join(folder_path, 'processed_test.h5'), 'w') as f:
        f.create_dataset('text_embeddings', data = x_eval)
        f.create_dataset('labels', data = y_test)

    x_eval = None
    for i in range(0, len(x_train), 4096):
        k = model.encode(x_train[i:i+4096])
        x_eval = k if x_eval is None else np.concatenate((x_eval, k), axis=0)
        print(f"train {i}/{len(x_train)}")
        
    print(x_eval.shape, len(y_train))

    with h5py.File(os.path.join(folder_path, 'processed_train.h5'), 'w') as f:
        f.create_dataset('text_embeddings', data = x_eval)
        f.create_dataset('labels', data = y_train)

def get_dataset(dataset, data_path):
    if dataset == 'dummy':
        embedding_size = 728
        max_sentence_len = 25
        
        num_classes = 2
        class_names = ['negative', 'positive']

        train_data = torch.cat((torch.randn(size = (5000, max_sentence_len, embedding_size)) - 1, torch.randn(size = (5000, max_sentence_len, embedding_size)) + 1), dim=0)
        test_data = torch.cat((torch.randn(size = (200, max_sentence_len, embedding_size)) - 1, torch.randn(size = (200, max_sentence_len, embedding_size)) + 1), dim=0)

        train_labels = torch.cat((torch.zeros(5000,), torch.ones(5000,)), dim = 0).long()
        test_labels = torch.cat((torch.zeros(200,), torch.ones(200,)), dim = 0).long()

        dst_train = TensorDataset(train_data, train_labels)
        dst_test = TensorDataset(test_data, test_labels)
    
    elif dataset in SST_DATASETS:
        if "SST2" in dataset:
            train_df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
            eval_df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/test.tsv', delimiter='\t', header=None)
            num_classes = 2
            class_names = ['negative', 'positive']

        elif "SST1" in dataset:
            train_df = pd.read_csv('https://raw.githubusercontent.com/gauravnuti/Text_Dataset_Condensation/master/data/SST1/train_SST1.tsv', delimiter='\t', header=None)
            eval_df = pd.read_csv('https://raw.githubusercontent.com/gauravnuti/Text_Dataset_Condensation/master/data/SST1/test_SST1.tsv', delimiter='\t', header=None)
            num_classes = 5
            class_names = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']

        max_sentence_len = 30

        text_col=train_df.columns.values[0] 
        category_col=train_df.columns.values[1]

        x_train = train_df[text_col].values.tolist()
        y_train = train_df[category_col].values.tolist()
        x_eval = eval_df[text_col].values.tolist()
        y_eval = eval_df[category_col].values.tolist()

        ### Word2Vec encoded vecors ###
        if 'w2v' in dataset:
            embed_model = api.load("word2vec-google-news-300")
            embedding_size = 300

            x_train = word_embedding_from_text_set(x_train, embed_model)
            x_eval = word_embedding_from_text_set(x_eval, embed_model)

        elif 'glove' in dataset:
            embed_model = api.load('glove-twitter-25')
            embedding_size = 25

            x_train = word_embedding_from_text_set(x_train, embed_model)
            x_eval = word_embedding_from_text_set(x_eval, embed_model)

        elif 'transformer' in dataset:
            st_model = 'paraphrase-mpnet-base-v2' #@param ['paraphrase-mpnet-base-v2', 'all-mpnet-base-v1', 'all-mpnet-base-v2', 'stsb-mpnet-base-v2', 'all-MiniLM-L12-v2', 'paraphrase-albert-small-v2', 'all-roberta-large-v1']
            model = SentenceTransformer(st_model)
            embedding_size = 768

            x_train = model.encode(x_train, output_value='token_embeddings')
            x_eval = model.encode(x_eval, output_value='token_embeddings') 

        
        print("Train size:", len(x_train))
        print("Test size:", len(x_eval))

        x_train_tensor = nn.utils.rnn.pad_sequence(x_train, batch_first=True)[:, :max_sentence_len, :]
        x_eval_tensor = nn.utils.rnn.pad_sequence(x_eval, batch_first=True)[:, :max_sentence_len, :]
        
        if 'flat' in dataset:
            x_train_tensor = x_train_tensor.reshape(-1, max_sentence_len * embedding_size)
            x_eval_tensor = x_eval_tensor.reshape(-1, max_sentence_len * embedding_size)

            embedding_size = max_sentence_len * embedding_size

        dst_train = TensorDataset(torch.Tensor(x_train_tensor),torch.Tensor(y_train).long()) 
        dst_test = TensorDataset(torch.Tensor(x_eval_tensor),torch.Tensor(y_eval).long()) 

    elif dataset in ['yahoo-flat', 'dbpedia-flat']:
        folder_path = os.path.join(os.environ['DATA_DIR'], dataset)

        train_data = h5py.File(os.path.join(folder_path, 'processed_train.h5'), 'r')
        test_data =  h5py.File(os.path.join(folder_path, 'processed_test.h5'), 'r')

        x_train = torch.tensor(np.array(train_data['text_embeddings']))
        y_train = torch.tensor(np.array(train_data['labels']))
        
        x_test = torch.tensor(np.array(test_data['text_embeddings']))
        y_test = torch.tensor(np.array(test_data['labels']))

        dst_train = TensorDataset(x_train, y_train.long())
        dst_test = TensorDataset(x_test, y_test.long())

        embedding_size = 768
        max_sentence_len = 1
        class_names =   [
                            "Society & Culture",
                            "Science & Mathematics",
                            "Health", "Education & Reference",
                            "Computers & Internet",
                            "Sports",
                            "Business & Finance",
                            "Entertainment & Music",
                            "Family & Relationships",
                            "Politics & Government"
                        ] if 'yahoo' in dataset else [
                            "Company",
                            "EducationalInstitution",
                            "Artist",
                            "Athlete",
                            "OfficeHolder",
                            "MeanOfTransportation",
                            "Building",
                            "NaturalPlace",
                            "Village",
                            "Animal",
                            "Plant",
                            "Album",
                            "Film",
                            "WrittenWork"
                        ]

        num_classes = len(class_names)

        print("Train size:", len(x_train))
        print("Test size:", len(x_test))

    else:
        exit('unknown dataset: %s'%dataset)

    #batch size changed form 256 to 16
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=True, drop_last=True)
    return embedding_size, max_sentence_len, num_classes, class_names, dst_train, dst_test, testloader

def word_embedding_from_text_set(train_set, vectors):
    sp = set(stopwords.words('english'))
    corpus = []
    for x in train_set:
        desc = x.lower()
        desc = re.sub('[^a-zA-Z]', ' ', desc)
        desc = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",desc)
        desc = re.sub("(\\d|\\W)+"," ",desc)
        corpus.append(word_tokenize(desc))

    embedded_train_set = []
    for i in range(len(corpus)):
        embedded_sentence = []
        for j in range(len(corpus[i])):
            # if corpus[i][j] in sp:
            #     continue
            if vectors.__contains__(corpus[i][j]):
                embedded_sentence.append(vectors[corpus[i][j]])
            else:
                embedded_sentence.append(vectors['unk'])
        if len(embedded_sentence) > 0:
            embedded_train_set.append(torch.tensor(np.array(embedded_sentence)))
        
    return embedded_train_set

class TensorDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]



def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling



def get_network(model, embed_dim, num_classes):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    # net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
    if model == 'MLP':
        net  = MLPV2(embed_dim = embed_dim, hidden_dim = 512, num_classes = num_classes)
    elif model == 'LSTMNet':
        net = LSTMNet(embed_dim = embed_dim, hidden_dim = 256, num_classes = num_classes)
    else:
        net = None
        exit('unknown model: %s'%model)

    gpu_num = torch.cuda.device_count()
    if gpu_num>0:
        device = 'cuda'
        if gpu_num>1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net



def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))



def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis



def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s'%args.dis_metric)

    return dis



def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop



def epoch(mode, dataloader, net, optimizer, criterion, args):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        train_data = datum[0].float().to(args.device)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output = net(train_data)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg



def evaluate_synset(it_eval, net, data_train, labels_train, testloader, args):
    net = net.to(args.device)
    data_train = data_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(data_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    for ep in range(Epoch+1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    return net, acc_train, acc_test



def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images



def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']: # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'B':  # multiple architectures with BatchNorm for DM experiments
        model_eval_pool = ['ConvNetBN', 'ConvNetASwishBN', 'AlexNetBN', 'VGG11BN', 'ResNet18BN']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL', 'ConvNetASwish']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        if 'BN' in model:
            print('Attention: Here I will replace BN with IN in evaluation, as the synthetic set is too small to measure BN hyper-parameters.')
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'SS':  # itself
        model_eval_pool = [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if strategy == 'None' or strategy == 'none' or strategy == '':
        return x

    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese: # Siamese augmentation:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0]
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}


if __name__ == '__main__':
    embedding_size, max_sentence_len, num_classes, class_names, dst_train, dst_test, testloader = get_dataset('SST1-transformer-flat', "")
    # encode_data('dbpedia-flat')
    print(torch.sum(dst_train.labels==0))
    print(torch.sum(dst_train.labels==1))
    print(dst_train.labels.shape)