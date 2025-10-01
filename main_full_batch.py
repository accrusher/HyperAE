import numpy as np
from tqdm import tqdm
import torch
import random
import os

from utils import (
    build_args,
    create_optimizer,
    load_missing_graph_dataset,
)

from utils import cluster_probing_full_batch
from utils import create_scheduler
from models import build_model

def generate_missing_mask(missing_index, num_nodes, mask_ratio=0.2):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[missing_index] = True
    known_index = torch.where(~mask)[0]
    num_mask = int(len(known_index) * mask_ratio)
    if num_mask > 0:
        random_mask = torch.randperm(len(known_index))[:num_mask]
        mask[known_index[random_mask]] = True
    return mask


def train(model, graph_adj,graph_hyper, feat, missing_index, optimizer, scheduler, max_epoch, device, num_classes, args):
    graph_adj = graph_adj.to(device)
    graph_hyper = graph_hyper.to(device)
    missing_mask= generate_missing_mask(missing_index, feat.shape[0], mask_ratio=0.20)
    x = feat.to(device)
    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss_APA, loss_DEG = model(graph_adj, graph_hyper, x, missing_mask, missing_index)
        loss = args.APA_para * loss_APA + args.DEG_para * loss_DEG

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")

        if  epoch==max_epoch-1 or epoch % 10 == 0:
            result_str = cluster_probing_full_batch(model, graph_adj, x, device)
            loss_log = {"loss_APA": args.APA_para * loss_APA, "loss_DEG": args.DEG_para * loss_DEG, "loss": loss}
            print(result_str)

    print("Final Results:")
    print(result_str)
    #with open('test.txt', 'a') as file:
    #    file.write(str(result_str)+'\n')
    return model


def main(args):
    device = f"{args.device}"

    dataset_name = args.dataset
    max_epoch = args.max_epoch
    
    optim_type = args.optimizer

    lr = args.lr
    weight_decay = args.weight_decay
    missing_rate = args.missing_rate

    graph_adj, graph_hyper, missing_index, (num_features, num_classes) = load_missing_graph_dataset(dataset_name, missing_rate,Hmask=args.hyperbuild)

    args.num_features = num_features
    args.num_nodes = graph_adj.ndata["feat"].size(0)
    
    model = build_model(args)
    model.to(device)
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)
    scheduler = create_scheduler(optimizer)

    x = graph_adj.ndata["feat"]
    labels = graph_adj.ndata["label"]

    train(model, graph_adj, graph_hyper,x, missing_index, optimizer,scheduler, max_epoch, device, num_classes, args)

    with torch.no_grad():
        embeddings = model.embed(graph_adj.to(device), x.to(device))
        embeddings = embeddings.cpu().numpy()
    
    labels = graph_adj.ndata["label"].numpy()



if __name__ == "__main__":
    print(f"torch.cuda.is_available()->{torch.cuda.is_available()}")
    args = build_args()


    print(f'{args.missing_rate}  {args.dataset}  {args.hyperbuild}',end=':')

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False      #introducing cnn optimization
    torch.backends.cudnn.deterministic = True   #using deterministic algorithms

    project_name = f""
    group_name = f""

    
    #print(args)

    main(args)

