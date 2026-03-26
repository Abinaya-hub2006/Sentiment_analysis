import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import kagglehub
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from model.lstm_model import SentimentLSTM
from utils.preprocess import clean_text, build_vocab, text_to_sequence

# -----------------------------
# SETTINGS
# -----------------------------
EPOCHS = 10
BATCH_SIZE = 64
VOCAB_SIZE = 5000
SAMPLES_PER_CLASS = 3000   # balanced dataset

# -----------------------------
# DOWNLOAD DATASET
# -----------------------------
print("📥 Downloading dataset...")
path = kagglehub.dataset_download("charunisa/chatgpt-sentiment-analysis")

print("📂 Loading dataset...")
files = os.listdir(path)
file_path = os.path.join(path, files[0])

df = pd.read_csv(file_path)

# -----------------------------
# CLEAN DATA
# -----------------------------
print("🧹 Cleaning dataset...")

# 🔥 CORRECT COLUMN FIX
df = df[['tweets', 'labels']]
df.columns = ['text', 'label']

# lowercase labels
df['label'] = df['label'].str.lower()

# remove nulls
df = df.dropna()

# debug check
print("Labels:", df['label'].unique())

# -----------------------------
# BALANCE DATASET
# -----------------------------
print("⚖️ Balancing dataset...")

df = df.groupby('label').sample(SAMPLES_PER_CLASS, random_state=42)

# -----------------------------
# PREPROCESS TEXT
# -----------------------------
df['text'] = df['text'].apply(clean_text)

texts = df['text'].tolist()

print("🧠 Building vocabulary...")
word_index = build_vocab(texts, max_words=VOCAB_SIZE)
joblib.dump(word_index, "model/word_index.pkl")

# convert text → sequence
X = [text_to_sequence(text, word_index, vocab_size=VOCAB_SIZE) for text in texts]
X = torch.tensor(X, dtype=torch.long)

# encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])
joblib.dump(encoder, "model/label_encoder.pkl")

y = torch.tensor(y, dtype=torch.long)

# -----------------------------
# TRAIN-VALIDATION SPLIT
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# -----------------------------
# MODEL
# -----------------------------
print("🚀 Initializing model...")
model = SentimentLSTM(vocab_size=VOCAB_SIZE + 1)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# -----------------------------
# RESUME CHECKPOINT
# -----------------------------
start_epoch = 0

if os.path.exists("model/checkpoint.pth"):
    checkpoint = torch.load("model/checkpoint.pth")
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"🔄 Resuming from epoch {start_epoch}")
else:
    print("🚀 Starting fresh training")

# -----------------------------
# TRAINING LOOP
# -----------------------------
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)

    # -----------------------------
    # VALIDATION
    # -----------------------------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)

            val_correct += (predicted == batch_y).sum().item()
            val_total += batch_y.size(0)

    val_acc = val_correct / val_total

    print(f"\n📊 Epoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    # -----------------------------
    # SAVE CHECKPOINT
    # -----------------------------
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, "model/checkpoint.pth")

# -----------------------------
# FINAL SAVE
# -----------------------------
torch.save(model.state_dict(), "model/model.pth")

print("\n🎉 Training Completed!")
print("📁 Saved: model.pth, word_index.pkl, label_encoder.pkl")