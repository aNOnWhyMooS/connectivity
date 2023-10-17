import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # print(tensor.numel())
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

def assign_pars(vector, model):
    new_pars = unflatten_like(vector, model.parameters())
    for old, new in zip(model.parameters(), new_pars):
        old.data = new.to(old.device).data
    
    return


def eval(loader, model, criterion, binary=False, coeffs_t=None):
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    for i, input in enumerate(loader):
        input_ids = input['input_ids'][0].cuda()
        attention_mask = input['attention_mask'][0].cuda()
        if 'token_type_ids' in input.keys():
            type_ids = input['token_type_ids'][0].cuda()
        else:
            type_ids = None
        if 'label' in input.keys():
            target=input['label'].cuda()
        else:
            target = None

        with torch.no_grad():
            output = model(input_ids, attention_mask, type_ids, 
                           target, coeffs_t=coeffs_t)
            loss = output.loss

        pred = output.logits.max(1, keepdim=True)[1]
        true = input['label'].cuda()
        pred = pred.view_as(true)
        # print(pred)
        if binary:
            pred[pred != 1] = 3
            pred[pred == 1] = 0
            pred[pred == 3] = 1
            with torch.no_grad():
                logits = output.logits.cpu()
                x = torch.max(torch.concat((logits[:, 0].view(-1, 1), 
                                            logits[:, 2].view(-1, 1)), 1), 1).values
                logits = torch.concat((logits[:, 1].view(-1,1), x.view(-1,1)), 1)
                loss = criterion(logits.cuda(), true)
            
        loss_sum += loss.item() * input['input_ids'][0].size(0)
        correct += pred.eq(true).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }

def train_epoch(loader, model, criterion, optimizer):
    loss_sum = 0.0
    correct = 0.0

    model.train()
    
    for i, input in enumerate(loader):
        input_ids = input['input_ids'][0].cuda()
        attention_mask = input['attention_mask'][0].cuda()
        if 'token_type_ids' in input.keys():
            type_ids = input['token_type_ids'][0].cuda()
        else:
            type_ids = None
        if 'label' in input.keys():
            target=input['label'].cuda()
        else:
            target = None
        
        output = model(input_ids, attention_mask, type_ids, target)
        loss = output.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input_ids.size(0)
        # pred = output.data.max(1, keepdim=True)[1]
        pred = output.logits.max(1, keepdim=True)[1]
        true = input['label'].cuda()
        correct += pred.eq(true.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def train_epoch_volume(loader, model, criterion, optimizer, vol_reg,
                      nsample):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, input in enumerate(tqdm(loader)):
        
        acc_loss = 0.
        for _ in range(nsample):
                output = model(input)
                acc_loss = acc_loss + output.loss
        acc_loss.div(nsample)
        
        vol = model.total_volume()
        log_vol = (vol + 1e-4).log()
        
        loss = acc_loss - vol_reg * log_vol

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input['input_ids'][0].size(0)
        pred = output.logits.max(1, keepdim=True)[1]
        true = input['label'].cuda()
        correct += pred.eq(true.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def train_epoch_multi_sample(loader, model, criterion, 
                             optimizer, nsample):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, input in enumerate(tqdm(loader)):
        input_ids = input['input_ids'][0].cuda()
        attention_mask = input['attention_mask'][0].cuda()
        if 'token_type_ids' in input.keys():
            type_ids = input['token_type_ids'][0].cuda()
        else:
            type_ids = None
        if 'label' in input.keys():
            target=input['label'].cuda()
        else:
            target = None
        
        acc_loss = 0.
        for _ in range(nsample):
                output = model(input_ids, attention_mask, type_ids, target)
                acc_loss += output.loss
        acc_loss.div(nsample)
        
        loss = acc_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input['input_ids'][0].size(0)
        pred = output.logits.max(1, keepdim=True)[1]
        true = input['label'].cuda()
        correct += pred.eq(true.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def train_transformer_epoch(
        loader, model, criterion, optimizer, nsample, vol_reg=1e-5, gradient_accumulation_steps=1, wandb=None
):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, input in enumerate(tqdm(loader)):
        # if i % 20 == 0:
        #     print(i, "batches completed")
        torch.cuda.empty_cache()

        input_ids = input['input_ids'][0].cuda()
        attention_mask = input['attention_mask'][0].cuda()
        if 'token_type_ids' in input.keys():
            type_ids = input['token_type_ids'][0].cuda()
        else:
            type_ids = None
        if 'label' in input.keys():
            target=input['label'].cuda()
        else:
            target = None
        
        loss = 0.
        for j in range(nsample):
            output = model(input_ids, attention_mask, type_ids, target)
            loss += output.loss
        loss.div(nsample)
        
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        vol = model.total_volume()
        log_vol = vol_reg * (vol + 1e-4).log()
        loss = loss - log_vol

        # optimizer.zero_grad()
        loss.backward()
        
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        loss_sum += loss.item() * input['input_ids'][0].size(0)
        pred = output.logits.max(1, keepdim=True)[1]
        true = input['label'].cuda()
        num_correct = pred.eq(true.view_as(pred)).sum().item()
        correct += num_correct

        if wandb and i%20 == 0:
            wandb.log({'simplex_loss':loss.item(),
                       'num_correct':correct,
                       'volume':vol})
            wandb.watch(model)

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }
