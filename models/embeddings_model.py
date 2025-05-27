from transformers import PreTrainedTokenizerFast
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from config import PROJECT_ROOT
from load_data import get_data, get_data_iter
import sys # Add sys import for sys.exit

class ArabicTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(
            text, 
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encodings.input_ids.squeeze(0),
            "attention_mask": encodings.attention_mask.squeeze(0),
        }
    

class ArabicEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=4, num_heads=4, max_position_embeddings=1024):
        super().__init__()
        self.embeddings = nn.Embedding(tokenizer.vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(hidden_size, tokenizer.vocab_size)

    def forward(self, input_ids, attention_mask):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        embeddings = self.embeddings(input_ids) + self.position_embeddings(position_ids)
        
        encoded = self.encoder(embeddings, src_key_padding_mask=~attention_mask.bool())
        
        logits = self.projection(encoded)
        return logits
    

if __name__ == "__main__":
    """ Load the data """
    texts = get_data_iter(path="data1.jsonl")
    arabic_texts = [text for text in texts]
    if not arabic_texts:
        print("Error: No texts loaded from data1.jsonl. Please check the file path and content.")
        sys.exit(1) # Exit if no data is loaded
    print(f"Loaded {len(arabic_texts)} texts.")
    """ Load the tokenizer """
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{PROJECT_ROOT}/outputs/custom_tokenizer.json")
    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'mask_token': '[MASK]'
    })
    """ Train the model """
    # Hyperparameters
    batch_size = 8
    learning_rate = 2e-4
    epochs = 10

    # Dataset & Loader
    dataset = ArabicTextDataset(arabic_texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    vocab_size = tokenizer.vocab_size
    model = ArabicEmbeddingModel(vocab_size)
    model.to(device)  # Move model to the selected device
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)  # Move input tensors to device
            attention_mask = batch["attention_mask"].to(device)  # Move attention mask to device

            # Simulate masked language modeling
            labels = input_ids.clone()
            # Randomly mask 15% tokens
            mask_indices = (torch.rand(input_ids.shape, device=device) < 0.15) & (input_ids != tokenizer.pad_token_id) # Generate mask on the correct device
            input_ids[mask_indices] = tokenizer.convert_tokens_to_ids("[MASK]")

            outputs = model(input_ids, attention_mask)
            logits = outputs

            # Only calculate loss for masked positions
            masked_logits = logits[mask_indices]
            masked_labels = labels[mask_indices]

            if masked_labels.nelement() > 0:
                loss = loss_fn(masked_logits.view(-1, vocab_size), masked_labels.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        tqdm.write(f"Epoch {epoch+1} Average Loss: {avg_loss}")
    torch.save(model.state_dict(), f"{PROJECT_ROOT}/outputs/embeddings_model.pt")