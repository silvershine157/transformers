from run_translation import Type1Discriminator
from transformers import AutoTokenizer
from datasets import load_dataset
import os

from torch.utils.data import DataLoader, Dataset
import random
import torch
import torch.nn as nn
import tqdm

device = torch.device("cuda")

def test1():
    model = Type1Discriminator()
    wmt16_ds = load_dataset('wmt16', 'ro-en')
    wmt16_train = wmt16_ds['train']
    tokenizer = AutoTokenizer.from_pretrained(
        'facebook/mbart-large-en-ro',
        cache_dir=None,
        use_fast=True,
        revision='main',
        use_auth_token=None,
    )
    tokenizer.tgt_lang = 'ro_RO'
    max_target_length = 128
    padding=False
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
    label_ids = labels["input_ids"]

def get_tokenizer(model_str='facebook/mbart-large-en-ro', tgt_lang='ro_RO'):
    tokenizer = AutoTokenizer.from_pretrained(
        model_str,
        cache_dir=None,
        use_fast=True,
        revision='main',
        use_auth_token=None,
    )
    tokenizer.tgt_lang = tgt_lang
    return tokenizer

def process_batch(batch, tokenizer):
    max_target_length = 128
    padding=True
    with tokenizer.as_target_tokenizer():
        target = tokenizer(batch['target'], max_length=max_target_length, padding=padding, truncation=True, return_tensors='pt').to(device)
        target_id = target.input_ids
        target_mask = target.attention_mask
        hypo = tokenizer(batch['hypo'], max_length=max_target_length, padding=padding, truncation=True, return_tensors='pt').to(device)
        hypo_id = hypo.input_ids
        hypo_mask = hypo.attention_mask
        # all [B, L]
    return target_id, target_mask, hypo_id, hypo_mask


class DiscrimDataset(Dataset):
    def __init__(self, mismatch_pairs):
        self.mismatch_pairs = mismatch_pairs
        
    def __len__(self):
        return len(self.mismatch_pairs)

    def __getitem__(self, idx):
        return self.mismatch_pairs[idx]

def get_data_loaders(data_dir, batch_size):

    target_path = os.path.join(data_dir, 'target.txt')
    hypo_path = os.path.join(data_dir, 'hypothesis.txt')
    with open(target_path, 'r', encoding='utf-8') as target_f:
        targets = target_f.readlines()
    with open(hypo_path, 'r', encoding='utf-8') as hypo_f:
        hypos = hypo_f.readlines()

    total_cnt = len(targets)
    assert total_cnt == len(hypos)
    exact_match_cnt = 0
    mismatch_pairs = []
    for target, hypo in zip(targets, hypos):
        if target == hypo:
            exact_match_cnt += 1
        else:
            item = {
                "target": target,
                "hypo": hypo
            }
            mismatch_pairs.append(item)
    print('total: ', total_cnt)

    train_ratio = 0.9
    total_size = len(mismatch_pairs)
    train_size = int(train_ratio*total_size)
    random.shuffle(mismatch_pairs)
    train_ds = DiscrimDataset(mismatch_pairs[:train_size])
    test_ds = DiscrimDataset(mismatch_pairs[train_size:])
    print('train size: ', len(train_ds))
    print('test size: ', len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, test_loader

def apply_disc(model, id_, mask, real):
    # loss
    # token acc
    # sentence acc

    dec_start_id = 250020 # in mbart config
    B, L = id_.shape
    start_ids = torch.tensor([dec_start_id], device=device).unsqueeze(1).expand(B, -1)
    _, state = model(ids=start_ids, state=None)

    score_list = []
    for l in range(L):
        id_step = id_[:, l].unsqueeze(1)
        score, state = model(ids=id_step, state=state)
        score_list.append(score)
    scores = torch.stack(score_list, dim=1)
    if not real:
        scores = -scores
    last_indices = mask.sum(dim=1, keepdim=True) - 1
    last_scores = torch.gather(scores, dim=1, index=last_indices).squeeze(1)

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    bce = criterion(scores, torch.ones_like(scores, dtype=torch.float32))
    len_factor = mask.float().mean(dim=1)
    loss = ((mask*bce).mean(dim=1)/len_factor).mean(dim=0)

    n_sent = B
    n_sent_correct = (last_scores > 0).sum().item()
    n_tok = mask.sum().item()
    n_tok_correct = (mask*(scores > 0)).sum().item()

    return loss, n_sent, n_sent_correct, n_tok, n_tok_correct

def train_epoch(loader, model, optimizer, tokenizer):
    running_loss = 0.0
    n_sent = 0
    n_sent_correct = 0
    n_tok = 0
    n_tok_correct = 0
    model.train()
    for batch in loader:
        target_id, target_mask, hypo_id, hypo_mask = process_batch(batch, tokenizer)
        loss1, ns1, nsc1, nt1, ntc1 = apply_disc(model, target_id, target_mask, True)
        loss0, ns0, nsc0, nt0, ntc0 = apply_disc(model, hypo_id, hypo_mask, False)
        loss = loss0 + loss1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n_sent += ns1 + ns0
        n_sent_correct += nsc1 + nsc0
        n_tok += nt1 + nt0
        n_tok_correct += ntc1 + ntc0
        running_loss += ns0*loss0.item() + ns1*loss1.item()
    avg_loss = running_loss/n_sent
    sent_acc = n_sent_correct/n_sent
    tok_acc = n_tok_correct/n_tok
    return avg_loss, sent_acc, tok_acc

def eval_epoch(loader, model, tokenizer):
    running_loss = 0.0
    n_sent = 0
    n_sent_correct = 0
    n_tok = 0
    n_tok_correct = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            target_id, target_mask, hypo_id, hypo_mask = process_batch(batch, tokenizer)
            loss1, ns1, nsc1, nt1, ntc1 = apply_disc(model, target_id, target_mask, True)
            loss0, ns0, nsc0, nt0, ntc0 = apply_disc(model, hypo_id, hypo_mask, False)
            n_sent += ns1 + ns0
            n_sent_correct += nsc1 + nsc0
            n_tok += nt1 + nt0
            n_tok_correct += ntc1 + ntc0
            running_loss += ns0*loss0.item() + ns1*loss1.item()
    avg_loss = running_loss/n_sent
    sent_acc = n_sent_correct/n_sent
    tok_acc = n_tok_correct/n_tok
    return avg_loss, sent_acc, tok_acc


def test2():
    #data_dir = 'data/train10000/'
    #data_dir = 'data/ro-en-train1000/'
    data_dir = 'data/ro-en-train100000'
    print("DATA: ", data_dir)
    ckpt_path = os.path.join(data_dir, 'type1_disc.pt')

    #tokenizer = get_tokenizer(model_str='facebook/mbart-large-en-ro', tgt_lang='ro_RO')
    #(model_str, tgt_lang) = ('facebook/mbart-large-en-ro', 'ro_RO')
    (model_str, tgt_lang) = ('facebook/mbart-large-50-many-to-many-mmt', 'en_XX')
    tokenizer = get_tokenizer(model_str=model_str, tgt_lang=tgt_lang)
    print("MODEL: ", model_str)
    print("TARGET LANG: ", tgt_lang)

    train_loader, test_loader = get_data_loaders(data_dir, 16)
    model = Type1Discriminator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 100
    best_tok_acc = 0.0
    for epoch in range(1, n_epochs+1):
        print("[epoch {:d}]".format(epoch))
        train_loss, train_sent_acc, train_tok_acc = train_epoch(train_loader, model, optimizer, tokenizer)
        print('[train] loss: {:.3f} sent acc: {:.3f} tok acc: {:.3f}'.format(train_loss, train_sent_acc, train_tok_acc))
        test_loss, test_sent_acc, test_tok_acc = train_epoch(test_loader, model, optimizer, tokenizer)
        print('[test] loss: {:.3f} sent acc: {:.3f} tok acc: {:.3f}'.format(test_loss, test_sent_acc, test_tok_acc))
        if test_tok_acc > best_tok_acc:
            best_tok_acc = test_tok_acc
            torch.save(model.state_dict(), ckpt_path)
            print("saved model", ckpt_path)


if __name__=='__main__':
    test2()