import numpy as np, argparse, time
import torch
from torch.utils.data import DataLoader
from dataloader import IEMOCAPRobertaCometDataset
from model import MaskedNLLLoss
from commonsense_model import CommonsenseGRUModel
from sklearn.metrics import f1_score, accuracy_score

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

def infer(model, loss_function, dataloader):
    losses, preds, labels, masks, losses_sense  = [], [], [], [], []
    alphas, alphas_f, alphas_b, vids = [], [], [], []
    max_sequence_len = []

    model.eval()

    for data in dataloader:
        
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

        s_ids = data[-1]
        lb_mapping = {'happy':0, 'sad':1, 'neutral':2, 'angry':3, 'excited':4, 'frustrated':5}
        label_mapping = ['']*len(lb_mapping)
        for e,v in lb_mapping.items():
            label_mapping[v] = e
        conversation_map = [["person {}: ".format(torch.argmax(qmask.transpose(0,1)[i][j])+1) + s + " [u{}, {}]".format(j+1, label_mapping[label[i][j]]) 
                                for j, s in enumerate(dataloader.dataset.sentences[s_id])
                            ] for i, s_id in enumerate(s_ids)]

        log_prob, _, alpha, alpha_f, alpha_b, _ = model(r1, r2, r3, r4, x5, x6, x1, o2, o3, qmask, umask, att2=True, additional_info= {'sentences': conversation_map})

        lp_ = log_prob.transpose(0,1).contiguous().view(-1, log_prob.size()[2]) # batch*seq_len, n_classes
        labels_ = label.view(-1) # batch*seq_len
        loss = loss_function(lp_, labels_, umask)

        pred_ = torch.argmax(lp_,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())
 
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

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)
    avg_sense_loss = round(np.sum(losses_sense)/np.sum(masks), 4)

    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels,preds, sample_weight=masks, average='weighted')*100, 2)
    
    return avg_loss, avg_accuracy, labels, preds, masks, [avg_fscore], [alphas, alphas_f, alphas_b, vids]


if __name__ == '__main__':

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
    parser.add_argument('--no-self-attn-emotions', action='store_true', default=False, help='use self attn in emotions chain')
    parser.add_argument('--model_path', default='./COSMIC/erc-training/models/best_fscore2.pt', help='path of the pretrained model')
    parser.add_argument('--print-self-attn', action='store_true', default=False, help='reverse flag print attention score')

    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    emo_gru = True
    n_classes  = 6
    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size

    global  D_s

    D_m = 1024
    D_s = 768
    D_g = 150
    D_p = 150
    D_r = 150
    D_i = 150
    D_h = 100
    D_a = 100

    D_e = D_p + D_r + D_i
    
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
    if args.model_path is not None and len(args.model_path) > 0:
        model = torch.load(open(args.model_path, 'rb'))
        if not hasattr(model.args, 'print_self_attn'):
            model.args.print_self_attn = args.print_self_attn
    
    print ('IEMOCAP COSMIC Model.')


    if cuda:
        model.cuda()

    loss_weights = torch.FloatTensor([1/0.086747,
                                      1/0.144406,
                                      1/0.227883,
                                      1/0.160585,
                                      1/0.127711,
                                      1/0.252668])

    if args.class_weight:
        loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()


    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(batch_size=batch_size,
                                                                  num_workers=0)

    test_fscores, test_losses = [], []
    best_loss, best_label, best_pred, best_mask = None, None, None, None

    start_time = time.time()
    # for data in test_loader:
    #     print(data)
    #     # break 
    #     # print(1/0)

    test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, attentions = infer(model, loss_function, test_loader)
        
    test_losses.append(test_loss)
    test_fscores.append(test_fscore)
    
    x = 'test-eval:   test_loss: {}, acc: {}, fscore: {}, time: {} sec'.format(test_loss, test_acc, test_fscore, round(time.time()-start_time, 2))
    
    print(x)

               
        

    