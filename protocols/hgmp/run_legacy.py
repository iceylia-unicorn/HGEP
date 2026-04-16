from protocols.common.paths import ARTIFACT_ROOT
from protocols.hgmp.utils_legacy import seed_everything

import torch.nn.functional as F
import os
from protocols.common.early_stop import EarlyStopping
from protocols.hgmp.pretrain_legacy import PreTrain
from protocols.hgmp.utils_legacy import mkdir, load_data4pretrain,graph_pool,new_load_data4pretrain,load_data4pretrain_metapath,set_params
from protocols.hgmp.prompt_legacy import HGNN, HeteroPrompt,GCL_GCN,GAT
from torch import nn, optim
from protocols.hgmp.data_legacy import DATA_ROOT, multi_class_NIG
import torch
import dgl
from protocols.hgmp.eval_legacy import acc_f1_over_batches,valid_over_batches
import argparse
import numpy as np

from pathlib import Path

PRETRAIN_DIR = Path("artifacts/checkpoints/hgmp/pretrain")
DOWNSTREAM_DIR = Path("artifacts/checkpoints/hgmp/downstream")
PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
DOWNSTREAM_DIR.mkdir(parents=True, exist_ok=True)
# this file can not move in ProG.utils.py because it will cause self-loop import
def model_create(input_dims,dataname, hgnn_type,pre_method, num_class,metadata,ntypes, task_type='multi_class_classification', tune_answer=False,args=None,num_etypes=None):
    if task_type in ['multi_class_classification', 'regression']:
        hid_dim =args.hidden_dim
        lr, wd = args.prompt_lr, args.weight_decay
        num_layer,num_heads,dropout=args.num_layers,args.num_heads,args.dropout

        # load pre-trained HGNN

        if(hgnn_type=='GCN'):
            hgnn=GCL_GCN(None,input_dims,hid_dim,num_class,num_layer,F.elu,dropout,hgnn_type)
        elif(hgnn_type=='GAT'):
            heads = [args.num_heads] * args.num_layers + [1]
            hgnn = GAT(None, input_dims, args.hidden_dim, num_class, args.num_layers, heads, F.elu, args.dropout,args.dropout, 0.05, False,hgnn_type)
        else:
            hgnn=HGNN(hid_dim=hid_dim,out_dim=hid_dim,hgnn_type=hgnn_type,num_layer=num_layer,num_heads=num_heads,dropout=dropout,metadata=metadata,ntypes=ntypes,num_etypes=num_etypes,input_dims=input_dims,args=args)
        pre_train_path = '{}/{}.{}.{}.hid{}.np{}.pth'.format(PRETRAIN_DIR, dataname, pre_method, hgnn_type, hid_dim, args.num_samples)
        hgnn.load_state_dict(torch.load(pre_train_path))
        #print("successfully load pre-trained weights for hgnn! @ {}".format(pre_train_path))
        for p in hgnn.parameters():
            p.requires_grad = False

        if tune_answer:
            PG = HeteroPrompt(token_dims=input_dims,ntypes=ntypes)

        opi = optim.Adam(filter(lambda p: p.requires_grad, PG.parameters()),
                         lr=lr,
                         weight_decay=wd)

        if(dataname=='IMDB'and args.classification_type!='GIG'):
            task_type='multi'

        if task_type == 'multi':
            lossfn = nn.BCEWithLogitsLoss()
        else:
            lossfn = nn.CrossEntropyLoss(reduction='mean')

        if tune_answer:
            if task_type == 'regression':
                answering = torch.nn.Sequential(
                    torch.nn.Linear(hid_dim, num_class),
                    torch.nn.Sigmoid())
            else:
                answering = torch.nn.Sequential(
                    torch.nn.Linear(hid_dim, num_class),
                    torch.nn.Softmax(dim=1))

            opi_answer = optim.Adam(filter(lambda p: p.requires_grad, answering.parameters()), lr=args.head_lr,
                                    weight_decay=args.weight_decay)
        hgnn.to(args.device)
        PG.to(args.device)
        return hgnn, PG, opi, lossfn, answering, opi_answer
    else:
        raise ValueError("model_create function hasn't supported {} task".format(task_type))


def pretrain(args, epoch_callback=None):
    # mkdir('./pre_trained_hgnn/')

    #print("load data...")
    batch_size=64
    num_sample=args.num_samples
    graph_list, in_dims, num_class = load_data4pretrain(args.feats_type, args.device, args.dataset,batch_size,num_sample)

    #print("create PreTrain instance...")
    graph=graph_list[0]
    metadata, ntypes = graph.canonical_etypes, graph.ntypes
    num_etypes=len(metadata)+1
    metadata = (ntypes, metadata)
    num_class=args.num_class
    # metadata=graph.metadata()
    # ntypes=graph.node_types
    # num_etypes=len(metadata[1])+1

    pt = PreTrain(ntypes=ntypes, args=args,metadata=metadata , num_class=num_class,num_etypes=num_etypes,input_dims=in_dims)

    pt.model.to(args.device)
    #print("pre-training...")


    pt.train(dataname=args.dataset, graph_list=graph_list, graph_batch_size=10, lr=args.pre_lr, decay=0.0001, epochs=args.pre_epoch,aug1='maskN', aug2='permE',node_batch_size=batch_size,seed=args.seed,aug_ration=args.aug_ration, epoch_callback=epoch_callback)

def train_one_outer_epoch(targetnode, epoch, train_loader, opi, lossfn, hgnn, PG, answering, classification_type, args):
    for j in range(1, epoch + 1):
        running_loss = 0.

        for batch_id, train_batch in enumerate(train_loader):  # bar2
            # print(train_batch)
            if(classification_type=='NIG'):
                batched_graph,_,batched_label=train_batch
            else:
                batched_graph,batched_label = train_batch

            batched_label=batched_label.to(args.device)
            batched_graph=batched_graph.to(args.device)


            prompted_graph = PG(batched_graph)


            x_dict=prompted_graph.ndata['x']
            edge_index_dict={}
            for etype in prompted_graph.canonical_etypes:
                edge_index = prompted_graph.edges(etype=etype)
                src = edge_index[0].unsqueeze(0)
                tar = edge_index[1].unsqueeze(0)
                edge_index = torch.cat((src, tar), dim=0)
                edge_index_dict[etype] = edge_index

            if(hgnn.hgnn_type=='HGT'):
                node_emb = hgnn(targetnode,x_dict,edge_index_dict)
            elif(hgnn.hgnn_type=='SHGN'):
                node_emb = hgnn(targetnode, prompted_graph,x_dict)
            elif (hgnn.hgnn_type == 'GCN'):
                node_emb = hgnn(prompted_graph,x_dict)
            elif (hgnn.hgnn_type == 'GAT'):
                node_emb = hgnn(prompted_graph,x_dict,False)


            graph_emb=graph_pool('mean',node_emb,prompted_graph)
            pre = answering(graph_emb)
            train_loss = lossfn(pre, batched_label)

            # print('\t\t==> answer_epoch {}/{} | batch {} | loss: {:.8f}'.format(j, answer_epoch, batch_id,
            #                                                                     train_loss.item()))

            opi.zero_grad()
            train_loss.backward()
            opi.step()
            running_loss += train_loss.item()

            # if batch_id % 5 == 4:  # report every 5 updates
            #     last_loss = running_loss / 5  # loss per batch
            #     # bar2.set_description('answer_epoch {}/{} | batch {} | loss: {:.8f}'.format(j, answer_epoch, batch_id,
            #     #                                                                     last_loss))
            #     print(
            #         'epoch {}/{} | batch {}/{} | loss: {:.8f}'.format(j, epoch, batch_id, len(train_loader), last_loss))
            #
            #     running_loss = 0.
        mean_loss=running_loss/len(train_loader)
        #print('training loss: {:.8f}'.format(mean_loss))



def prompt_w_h(dataname="IMDB", hgnn_type="HGT", num_class=5, task_type='multi_class_classification',pre_method=None,args=None):

    train_list, valid_list,test_list = multi_class_NIG(dataname, num_class, shots=args.shots,classification_type=args.classification_type,feats_type=args.feats_type)
    graph=train_list[0][0]
    metadata,ntypes=graph.canonical_etypes,graph.ntypes
    num_etypes=len(metadata)+1
    metadata=(ntypes,metadata)
    input_dims = []
    for nt in ntypes:
        scheme, dtype = graph.node_attr_schemes(nt)['x']
        input_dims.append(scheme[0])
    nodes=graph.ndata['y'].keys()
    for key in nodes:
        targetnode=key
    dataloader=dgl.dataloading.GraphDataLoader
    train_loader = dataloader(train_list, batch_size=10, shuffle=True)
    valid_loader = dataloader(valid_list, batch_size=10, shuffle=True)
    test_loader = dataloader(test_list, batch_size=10, shuffle=True)

    hgnn, PG, opi_pg, lossfn, answering, opi_answer = model_create(input_dims,dataname, hgnn_type,pre_method, num_class,metadata,ntypes,task_type, True,args,num_etypes)
    answering.to(args.device)

    # inspired by: Hou Y et al. MetaPrompting: Learning to Learn Better Prompts. COLING 2022
    # if we tune the answering function, we update answering and prompt alternately.
    # ignore the outer_epoch if you do not wish to tune any use any answering function
    # (such as a hand-crafted answering template as prompt_w_o_h)
    outer_epoch = args.prompt_epoch
    answer_epoch = 1  # 50
    prompt_epoch = 1  # 50

    # training stage
    PG_early_stopping=EarlyStopping(patience=30,verbose=False,
                                 save_path='{}/checkpoints/hgmp/downstream/checkpoint_PG_{}_{}.pth'.format(ARTIFACT_ROOT,args.dataset,args.hgnn_type))
    answer_early_stopping = EarlyStopping(patience=30, verbose=False,
                                   save_path='{}/checkpoints/hgmp/downstream/checkpoint_answer_{}_{}.pth'.format(ARTIFACT_ROOT,args.dataset, args.hgnn_type))
    for i in range(1, outer_epoch + 1):
        #print(("{}/{} frozen gnn | frozen prompt | *tune answering function...".format(i, outer_epoch)))
        # tune task head
        answering.train()
        PG.eval()
        train_one_outer_epoch(targetnode,answer_epoch, train_loader, opi_answer, lossfn, hgnn, PG, answering,args.classification_type,args)

        #print("{}/{}  frozen gnn | *tune prompt |frozen answering function...".format(i, outer_epoch))
        # tune prompt
        answering.eval()
        PG.train()
        train_one_outer_epoch(targetnode,prompt_epoch, train_loader, opi_pg, lossfn, hgnn, PG, answering,args.classification_type,args)

        # valid stage
        answering.eval()
        PG.eval()
        valid_loss=valid_over_batches(valid_loader, PG, hgnn, answering, num_class, task_type, device=args.device,targetnode=targetnode,dataname=dataname,lossfn=lossfn,classification_type=args.classification_type)
        answer_early_stopping(valid_loss,answering)
        PG_early_stopping(valid_loss, PG)
        if(answer_early_stopping.early_stop):
            #print('Early stopping!')
            break
    # testing stage
    PG.load_state_dict(torch.load('{}/checkpoints/hgmp/downstream/checkpoint_PG_{}_{}.pth'.format(ARTIFACT_ROOT,args.dataset,args.hgnn_type)))
    answering.load_state_dict(torch.load('{}/checkpoints/hgmp/downstream/checkpoint_answer_{}_{}.pth'.format(ARTIFACT_ROOT,args.dataset, args.hgnn_type)))

    acc,ma_f1=acc_f1_over_batches(test_loader, PG, hgnn, answering, num_class, task_type, device=args.device,targetnode=targetnode,dataname=dataname,classification_type=args.classification_type)
    return acc,ma_f1

if __name__ == '__main__':

    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats_type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '-1 - freebase undirected; ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                         '4 - only term features (id vec for others);' +
                         '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden_dim', type=int, default=512, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num_heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--pre_epoch', type=int, default=300, help='Number of pretraining epochs.')
    ap.add_argument('--prompt_epoch', type=int, default=300, help='Number of prompt training epochs.')
    ap.add_argument('--patience', type=int, default=7, help='Patience.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--prompt_lr', type=float, default=1e-3)
    ap.add_argument('--head_lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--slope', type=float, default=0.05)
    ap.add_argument('--dataset', type=str, default='DBLP')
    ap.add_argument('--edge_feats', type=int, default=64)
    ap.add_argument('--device', type=int, default=2)
    ap.add_argument('--schedule_step', type=int, default=300)
    ap.add_argument('--use_norm', type=bool, default=False)
    ap.add_argument('--pretext', type=str, default='GraphCL')
    ap.add_argument('--hgnn_type', type=str, default='GCN')
    ap.add_argument('--num_class', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--classification_type', type=str, default='NIG',help='NIG EIG GIG')
    ap.add_argument('--num_samples', type=int, default=100)
    ap.add_argument('--pre_lr', type=int, default=1e-3)
    ap.add_argument('--aug_ration', type=int, default=1e-3)


    args=ap.parse_args()
    seed_everything(args.seed)

    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        args.device = torch.device("cuda:"+str(args.device))
    else:
        print("CUDA is not available")
        args.device = torch.device("cpu")


    shots=[1,3,5]
    tasktype=['NIG','EIG','GIG']
    args.dataset = 'ACM'
    args.num_class = 3
    args = set_params(args)
    for shot in shots:
        args.shots=shot
        for type in tasktype:
            args.classification_type=type
            mif1_list=[]
            maf1_list=[]
            for i in range(3):
                mif1,maf1=prompt_w_h(dataname=args.dataset, hgnn_type=args.hgnn_type, num_class=args.num_class, task_type='multi_class_classification',pre_method='GraphCL',args=args)
                mif1_list.append(mif1)
            print(f'shot: {args.shots}, tasktype: {args.classification_type}')
            print(f'{mif1_list}')
