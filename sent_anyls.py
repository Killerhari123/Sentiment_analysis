import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

# Ensure nltk resources are downloaded
nltk.download('punkt')

# Sample dataset
sentences = [
    "I love this movie, it was fantastic!",
    "This film was terrible and boring.",
    "Absolutely wonderful experience, highly recommend!",
    "I did not like this movie at all.",
    "The plot was engaging, but the acting was bad."
]
labels = [1, 0, 1, 0, 0]  # 1: Positive, 0: Negative

# Tokenization
all_words = []
for sent in sentences:
    all_words.extend(word_tokenize(sent.lower()))

# Build vocabulary
word_counts = Counter(all_words)
vocab = {word: idx+1 for idx, (word, _) in enumerate(word_counts.items())}
vocab["<PAD>"] = 0

# Convert sentences to numerical data
def encode_sentence(sentence, vocab, max_len=10):
    tokens = word_tokenize(sentence.lower())
    encoded = [vocab.get(word, 0) for word in tokens]
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    return encoded[:max_len]

max_len = 10  # Fixed sequence length
encoded_sentences = np.array([encode_sentence(sent, vocab, max_len) for sent in sentences])
labels = np.array(labels)

# Convert to PyTorch tensors
X_tensor = torch.tensor(encoded_sentences, dtype=torch.long)
y_tensor = torch.tensor(labels, dtype=torch.float32)

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define Model
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=8, hidden_dim=16):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return torch.sigmoid(out)

# Initialize model, loss, and optimizer
vocab_size = len(vocab)
model = SentimentModel(vocab_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# Training loop
epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for X_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs.view(-1), y_batch.float().view(-1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Evaluate the model
model.eval()
print("\nTesting Model on Sample Inputs:")
test_sentences = [
    "This was an amazing movie!",
    "I hated every moment of it.",
    "It was an okay experience, not great but not terrible."
]
test_encoded = torch.tensor([encode_sentence(sent, vocab, max_len) for sent in test_sentences], dtype=torch.long)
predictions = model(test_encoded).squeeze().detach().numpy()

for i, sent in enumerate(test_sentences):
    print(f"Sentence: '{sent}' -> Sentiment Score: {predictions[i]:.4f}")
