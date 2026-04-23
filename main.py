import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import math
from torch.utils.data import TensorDataset, DataLoader


class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.seg_embed = nn.Embedding(2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, segment_label):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(segment_label)
        return self.dropout(self.norm(embedding))


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(self.dropout(weights), v)
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.out_linear(context)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(heads, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.attention(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class BERTForClassification(nn.Module):
    def __init__(self, vocab_size, hidden=256, n_layers=6, attn_heads=8, max_len=256):
        super().__init__()
        self.embedding = BERTEmbeddings(vocab_size, hidden, max_len)
        self.layers = nn.ModuleList([TransformerBlock(hidden, attn_heads) for _ in range(n_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 2)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, mask):
        seg_label = torch.zeros_like(x)
        x = self.embedding(x, seg_label)
        for layer in self.layers:
            x = layer(x, mask)
        input_mask_expanded = (x != 0).float()
        sum_embeddings = torch.sum(x, 1)
        sum_mask = x.size(1)
        return self.classifier(sum_embeddings / sum_mask)

def text_to_tensors(texts, word_to_idx, max_len=128):
    data = []
    for text in texts:
        ids = [word_to_idx.get(word, 1) for word in text.split()]
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [0] * (max_len - len(ids))
        data.append(ids)
    return torch.tensor(data, dtype=torch.long)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    with open('20news_processed.pkl', 'rb') as f:
        X_train, X_val, X_test, y_train, y_val, y_test, word_to_idx, vocab_size = pickle.load(f)

    max_seq_len = 256
    # 转换三个数据集为张量
    train_inputs = text_to_tensors(X_train, word_to_idx, max_len=max_seq_len)
    val_inputs = text_to_tensors(X_val, word_to_idx, max_len=max_seq_len)
    test_inputs = text_to_tensors(X_test, word_to_idx, max_len=max_seq_len)

    train_loader = DataLoader(TensorDataset(train_inputs, torch.tensor(y_train)), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_inputs, torch.tensor(y_val)), batch_size=16)
    test_loader = DataLoader(TensorDataset(test_inputs, torch.tensor(y_test)), batch_size=16)

    model = BERTForClassification(vocab_size=vocab_size, hidden=256, n_layers=6, max_len=max_seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = 5
    no_improve_epochs = 0
    max_epochs = 100
    model_save_path = 'best_bert_model.pth'

    print("开始训练...")
    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0
        for ids, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            mask = (ids > 0).unsqueeze(1).unsqueeze(2)

            optimizer.zero_grad()
            outputs = model(ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for ids, labels in val_loader:
                ids, labels = ids.to(device), labels.to(device)
                mask = (ids > 0).unsqueeze(1).unsqueeze(2)
                preds = model(ids, mask).argmax(dim=1)
                val_correct += (preds == labels).sum().item()

        current_val_acc = val_correct / len(y_val)
        avg_train_loss = total_train_loss / len(train_loader)

        print(f"Epoch {epoch + 1:2d} | Train Loss: {avg_train_loss:.4f} | Val Acc: {current_val_acc:.4f}")

        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"检测到更好的验证集效果，模型已保存。")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"验证集准确率已连续 {patience} 轮未提升，早停。")
            break

    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    test_correct = 0
    with torch.no_grad():
        for ids, labels in test_loader:
            ids, labels = ids.to(device), labels.to(device)
            mask = (ids > 0).unsqueeze(1).unsqueeze(2)
            preds = model(ids, mask).argmax(dim=1)
            test_correct += (preds == labels).sum().item()

    final_test_acc = test_correct / len(y_test)
    print(f"最佳模型在测试集上的准确率: {final_test_acc:.4f}")


if __name__ == "__main__":
    main()