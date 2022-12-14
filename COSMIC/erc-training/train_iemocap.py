import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dataloader import IEMOCAPRobertaCometDataset
from model import MaskedNLLLoss, config_arg
from commonsense_model import CommonsenseGRUModel
from sklearn.metrics import f1_score, accuracy_score
import os

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_IEMOCAP_loaders(batch_size=32, num_workers=0, pin_memory=False):
    trainset = IEMOCAPRobertaCometDataset('train')
    validset = IEMOCAPRobertaCometDataset('valid')
    testset = IEMOCAPRobertaCometDataset('test')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    # ==============
    # variables to save all the values of training process
    # ==============
    losses, preds, labels, masks, losses_sense  = [], [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval() ### for remove the affect of dropout in infer process 

    # ==============
    # seed function is used to get the same result between different runs.
    # ==============
    seed_everything(seed)

    # ==============
    # each iterator is a minibatch,
    # ==============
    for data in dataloader:  ### data variable will save all data in a minibatch 
        if train:
            optimizer.zero_grad()
        
        # ==============
        # LOAD DATA
        # ==============
        # check the order of variables in here to know the type of data. 
        # r1, r2, r3, r4 => self.roberta1, self.roberta2, self.roberta3, self.roberta4  is the hidden states of utterances encoded by 4 layer in pretrained Roberta.
        #                 => remember that, COSMIC framework using pretrained roberta for encoding source sentence without considering context information (check COSMIC paper).
        #            where vid (video id) is a sample id of conversation 
        # x1, ... x5, x6, => xIntent, ... xEffect, self.roberta3, xReact [vid]
        # ==============
        """
        torch.FloatTensor(self.roberta1[vid]),\
        torch.FloatTensor(self.roberta2[vid]),\
        torch.FloatTensor(self.roberta3[vid]),\
        torch.FloatTensor(self.roberta4[vid]),\
        torch.FloatTensor(self.xIntent[vid]),\
        torch.FloatTensor(self.xAttr[vid]),\
        torch.FloatTensor(self.xNeed[vid]),\
        torch.FloatTensor(self.xWant[vid]),\
        torch.FloatTensor(self.xEffect[vid]),\
        torch.FloatTensor(self.xReact[vid]),\
        torch.FloatTensor(self.oWant[vid]),\
        torch.FloatTensor(self.oEffect[vid]),\
        torch.FloatTensor(self.oReact[vid]),\
        torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.speakers[vid]]),\
        torch.FloatTensor([1]*len(self.labels[vid])),\
        torch.LongTensor(self.labels[vid]),\
        vid
        """
        r1, r2, r3, r4, \
        x1, x2, x3, x4, x5, x6, \
        o1, o2, o3, \
        qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        
        # ============== 
        # FORWARD data to model. 
        # ==============
        log_prob, _, alpha, alpha_f, alpha_b, _ = model(r1, r2, r3, r4, x5, x6, x1, o2, o3, qmask, umask, att2=True)

        # ============== 
        # compute LOSSES 
        # ==============
        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        # ============== 
        # logging
        # ==============
        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())

        if train:
            # ============== 
            # BACKWARD loss for updating weights
            # ==============
            total_loss = loss
            total_loss.backward()
            if args.tensorboard: ### for logging  
                for param in model.named_parameters():
                    if param[0] is not None and param[1].grad is not None:
                        writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], [], float('nan'),[]

    # ============== 
    # average loss, just consider the real data, without considering to padding samples. 
    # ==============
    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_sense_loss = round(np.sum(losses_sense)/np.sum(masks), 4)

    # ============== 
    # compute acc and f1 scores based on the prediction result returned in `preds` variable
    # ==============
    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)
    
    return avg_loss, avg_accuracy, labels, preds, masks, [avg_fscore], [alphas, alphas_f, alphas_b, vids]


if __name__ == '__main__':
    
    # ==============
    # hype-parameters (setting for training) of this program 
    # ==============
    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.25, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    parser.add_argument('--attention', default='general2', help='Attention type in context GRU')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--mode1', type=int, default=2, help='Roberta features to use')
    parser.add_argument('--seed', type=int, default=100, metavar='seed', help='seed')
    parser.add_argument('--norm', type=int, default=3, help='normalization strategy')
    parser.add_argument('--residual', action='store_true', default=False, help='use residual connection')
    config_arg(parser)

    args = parser.parse_args()
    print(args)


    # ==============
    # check cuda and run tensorboard (tensorboard is a library logging the learning curve values)
    # ==============
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(logdir="runs/iemocap:noselfemo-{}:seed-{}:dr-{}".format(args.no_self_attn_emotions, args.seed, args.dropout))


    # ==============
    # init some setting - from program input argurments => variables => training process.
    # ==============
    emo_gru = True
    n_classes  = 6
    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size

    # ==============
    # init some setting - from program input argurments => variables => training process.
    # ==============

    global  D_s # global variable, which can be used in a local function.


    # ==============
    # dimmenssions of features in the COSMIC framework, context, internal, external states. 
    # ==============
    D_m = 1024
    D_s = 768
    D_g = 150
    D_p = 150
    D_r = 150
    D_i = 150
    D_h = 100
    D_a = 100

    D_e = D_p + D_r + D_i

    global seed
    seed = args.seed
    seed_everything(seed)
    
    # ==============
    # init model structure
    # ==============
    model = CommonsenseGRUModel(D_m, D_s, D_g, D_p, D_r, D_i, D_e, D_h, D_a,
                                n_classes=n_classes,
                                listener_state=args.active_listener,
                                context_attention=args.attention,
                                dropout_rec=args.rec_dropout,
                                dropout=args.dropout,
                                emo_gru=emo_gru,
                                mode1=args.mode1,
                                norm=args.norm,
                                residual=args.residual,
                                args=args)

    print ('IEMOCAP COSMIC Model.')


    if cuda:
        model.cuda()

    
    # ==============
    # loss weights, which also is a predifined weight of weighted-F1 measurement metrics.
    # ==============
    loss_weights = torch.FloatTensor([1/0.086747,
                                      1/0.144406,
                                      1/0.227883,
                                      1/0.160585,
                                      1/0.127711,
                                      1/0.252668])

    # ==============
    # loss function, Negative Log Likelyhood loss 
    # ==============
    if args.class_weight:
        loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()

    # ==============
    # optimizer algorithms, support update learning rate via each epochs / steps
    # ==============
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    
    lf = open('logs/cosmic_iemocap_logs.txt', 'a')

    # ==============
    # load dataset.
    #    => the data loaded in here cotaining training data (utterances vector representation, labels)
    # ==============
    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(batch_size=batch_size,
                                                                  num_workers=0)

    valid_losses, valid_fscores = [], []
    test_fscores, test_losses = [], []
    best_loss, best_label, best_pred, best_mask = None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()

        # ==============
        # IMPORTANT
        # ==============
        # train model - run forward and backward using training set
        # 1 epoch, one time the model forward all training data, containing maybe 100 steps 
        #   (each step is one time the model is feeded 1 mini batch)
        # ==============
        train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, train_loader, e, optimizer, train=True)


        # ==============
        # eval model - run forward and backward using dev and test sets
        # ==============
        valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, valid_loader, e)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = train_or_eval_model(model, loss_function, test_loader, e)
            
        # ==============
        # logging training values
        # ==============
        valid_losses.append(valid_loss)
        valid_fscores.append(valid_fscore)
        test_losses.append(test_loss)
        test_fscores.append(test_fscore)
        
        if args.tensorboard:
            writer.add_scalar('valid/accuracy', valid_acc, e)
            writer.add_scalar('valid/loss', valid_loss, e)
            writer.add_scalar('valid/f1score', valid_fscore, e)

            writer.add_scalar('test/accuracy', test_acc, e)
            writer.add_scalar('test/loss', test_loss, e)
            writer.add_scalar('test/f1score', test_fscore, e)

            writer.add_scalar('train/accuracy', train_acc, e)
            writer.add_scalar('train/loss', train_loss, e)
            writer.add_scalar('train/f1score', train_fscore, e)
            
        x = 'epoch: {}, train_loss: {}, acc: {}, fscore: {}, valid_loss: {}, acc: {}, fscore: {}, test_loss: {}, acc: {}, fscore: {}, time: {} sec'.format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2))
        
        print (x)
        lf.write(x + '\n')


        # ==============
        # save best model founded. 
        # ==============
        def mkdir_fol():
            try:
                os.mkdir('models/')
            except OSError as error:
                print(error)  

        if valid_fscore >= valid_fscores[0][np.argmax(valid_fscores[0])]:
            mkdir_fol()
            torch.save(model, open('models/best_fscore.pt', 'wb'))
        if valid_loss <= valid_losses[np.argmin(valid_losses)]:
            mkdir_fol()
            torch.save(model, open('models/min_loss.pt', 'wb'))
               
    if args.tensorboard:
        writer.close()

    # ==============
    # eval model with best score, last time - when finished all training process.
    # ==============
    valid_fscores = np.array(valid_fscores).transpose()
    test_fscores = np.array(test_fscores).transpose()
    
    score1 = test_fscores[0][np.argmin(valid_losses)]
    score2 = test_fscores[0][np.argmax(valid_fscores[0])]
    scores = [score1, score2]
    scores = [str(item) for item in scores]
    
    print ('Test Scores: Weighted F1')
    print('@Best Valid Loss: {}'.format(score1))
    print('@Best Valid F1: {}'.format(score2))

    rf = open('results/cosmic_iemocap_results.txt', 'a')
    rf.write('\t'.join(scores) + '\t' + str(args) + '\n')
    rf.close()
    