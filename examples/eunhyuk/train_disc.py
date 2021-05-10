import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import tqdm

device = torch.device("cuda")

train_path = 'data/t5-train-150k/triplet_data.dat'
test_path = 'data/t5-test/triplet_data.dat'
TARGET_MAXLEN = 128


class T5PrefixDiscriminator(nn.Module):
    def __init__(self, base):
        super(T5PrefixDiscriminator, self).__init__()
        self.base = base
        self.out_layer = nn.Linear(512, 1)

    def forward(self, enc_ids, enc_mask, dec_ids, dec_mask):
        # for training
        base_out = self.base(
            input_ids=enc_ids,
            attention_mask=enc_mask,
            decoder_input_ids=dec_ids,
            decoder_attention_mask=dec_mask,
            output_hidden_states=True
        )
        dec_hidden = base_out.decoder_hidden_states[-1]
        scores = self.out_layer(dec_hidden).squeeze(2) # [B, L]
        return scores

    def prepare_incremental(self, input_ids, **kwargs):
        enc_out = self.base.encoder(input_ids, **kwargs)
        return enc_out

    def forward_incremental():
        pass


class ListDataset(Dataset):
    def __init__(self, datalist):
        self.datalist = datalist
    def __len__(self):
        return len(self.datalist)
    def __getitem__(self, idx):
        return self.datalist[idx]


def process_batch(batch, tokenizer):
    prefix = "summarize: "
    src = [prefix + s for s in batch["src"]]
    src_tokout = tokenizer(src, max_length=1024, padding="max_length", return_tensors="pt", truncation=True).to(device)
    with tokenizer.as_target_tokenizer():
        tgt_tokout = tokenizer(batch["tgt"], max_length=TARGET_MAXLEN, padding="max_length", return_tensors="pt", truncation=True).to(device)
        hypo_tokout = tokenizer(batch["hypo"], max_length=TARGET_MAXLEN, padding="max_length", return_tensors="pt", truncation=True).to(device)
    return src_tokout, tgt_tokout, hypo_tokout


def apply_disc_naive(model, src_tokout, tgt_tokout, hypo_tokout):
    real_scores = model(
        enc_ids=src_tokout.input_ids,
        enc_mask=src_tokout.attention_mask,
        dec_ids=tgt_tokout.input_ids,
        dec_mask=tgt_tokout.attention_mask,
    )
    fake_scores = model(
        enc_ids=src_tokout.input_ids,
        enc_mask=src_tokout.attention_mask,
        dec_ids=hypo_tokout.input_ids,
        dec_mask=hypo_tokout.attention_mask,
    )
    bce = nn.BCEWithLogitsLoss()
    B, L = real_scores.shape
    real_label = torch.ones(B, L).to(device)
    real_loss = bce(real_scores, real_label)
    fake_label = torch.zeros(B, L).to(device)
    fake_loss = bce(fake_scores, fake_label)
    total_loss = 0.5*(real_loss + fake_loss)
    correct = torch.cat([real_scores > 0.0, fake_scores < 0.0], dim=0)
    correct_cnt = correct.sum(dim=0)
    total_cnt = 2*B
    return total_loss, correct_cnt, total_cnt



def train_epoch(model, loader, optimizer, tokenizer, limit_iter=None):
    model.train()
    running_B = 0
    running_loss = 0.0
    running_correct = torch.zeros(TARGET_MAXLEN, dtype=torch.long, device=device)
    for batch_idx, batch in tqdm.tqdm(enumerate(loader)):
        src_tokout, tgt_tokout, hypo_tokout = process_batch(batch, tokenizer)
        total_loss, correct_cnt, total_cnt = apply_disc_naive(model, src_tokout, tgt_tokout, hypo_tokout)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        running_B += total_cnt
        running_correct += correct_cnt
        running_loss += total_cnt*total_loss.item()
        if limit_iter is not None and batch_idx >= limit_iter-1:
            break
    acc_per_pos = running_correct/running_B
    avg_loss = running_loss/running_B
    return avg_loss, acc_per_pos


def eval_epoch(model, loader, tokenizer, limit_iter=None):
    model.eval()
    running_B = 0
    running_loss = 0.0
    running_correct = torch.zeros(TARGET_MAXLEN, dtype=torch.long, device=device)
    with torch.no_grad():
        for batch_idx, batch in tqdm.tqdm(enumerate(loader)):
            src_tokout, tgt_tokout, hypo_tokout = process_batch(batch, tokenizer)
            total_loss, correct_cnt, total_cnt = apply_disc_naive(model, src_tokout, tgt_tokout, hypo_tokout)
            running_B += total_cnt
            running_correct += correct_cnt
            running_loss += total_cnt*total_loss.item()
            if limit_iter is not None and batch_idx >= limit_iter-1:
                break
    acc_per_pos = running_correct/running_B
    avg_loss = running_loss/running_B
    return avg_loss, acc_per_pos


def main():
    t5model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
    model = T5PrefixDiscriminator(base=t5model).to(device)
    train_triplets = torch.load(train_path)
    test_triplets = torch.load(test_path)
    tokenizer = AutoTokenizer.from_pretrained('tromedlov/t5-small-cnn')
    n_valid = 1000
    train_ds = ListDataset(train_triplets[n_valid:])
    valid_ds = ListDataset(train_triplets[:n_valid])
    test_ds = ListDataset(test_triplets)
    train_loader = DataLoader(train_ds, batch_size=4)
    valid_loader = DataLoader(valid_ds, batch_size=4)
    test_loader = DataLoader(test_ds, batch_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 100
    for epoch in range(n_epochs):
        print("epoch: ", epoch)
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, tokenizer, limit_iter=200)
        print("train loss: ", train_loss)
        print("train acc[-1]: ", train_acc[-1].item())
        vaild_loss, valid_acc = eval_epoch(model, valid_loader, tokenizer, limit_iter=200)
        print("valid loss: ", vaild_loss,)
        print("valid acc[-1]: ", valid_acc[-1].item())
        test_loss, test_acc = eval_epoch(model, test_loader, tokenizer, limit_iter=500)
        print("test loss: ", test_loss,)
        print("test acc[-1]: ", test_acc[-1].item())
        print("<test acc>")
        print(test_acc)


if __name__=='__main__':
    main()
