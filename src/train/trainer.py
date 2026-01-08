from src.utils.mask import create_mask

class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for words, tags in loader:
            words, tags = words.to(self.device), tags.to(self.device)
            mask = create_mask(words)

            loss = self.model(words, tags, mask)

            if self.class_weights is not None:
                loss = loss * self.class_weights.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)
