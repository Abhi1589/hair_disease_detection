import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data augmentation and normalization for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Reduced augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root=r'C:\Users\abhik\Downloads\Hair_Diseases\train', transform=train_transform)
val_dataset = datasets.ImageFolder(root=r'C:\Users\abhik\Downloads\Hair_Diseases\val', transform=val_test_transform)
test_dataset = datasets.ImageFolder(root=r'C:\Users\abhik\Downloads\Hair_Diseases\test', transform=val_test_transform)

# Create data loaders with batch size 16
train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Modified VGG model
class ModifiedVGG(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedVGG, self).__init__()
        # Load pretrained VGG16 model
        vgg = models.vgg16(pretrained=True)
        
        # Unfreeze more layers
        for param in vgg.features[:15].parameters():  # Unfreeze the first 15 layers
            param.requires_grad = True
            
        self.features = vgg.features
        
        # Modify classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),  # Increased dropout for regularization
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1)  # Use LogSoftmax for numerical stability
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Initialize model
num_classes = len(train_dataset.classes)
model = ModifiedVGG(num_classes).to(device)

# Training parameters
learning_rate = 0.0001  # Decreased learning rate
epochs = 20
criterion = nn.NLLLoss()  # Use NLLLoss with LogSoftmax
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay

# Lists for tracking metrics
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_acc = 0.0
patience = 3  # Early stopping patience
counter = 0

# Training loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    # Save best model and implement early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'vgg.pth')
        counter = 0  # Reset counter if improvement
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Test final model
model.load_state_dict(torch.load('vgg.pth'))
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Testing'):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_accuracy = test_correct / test_total
print(f'\nFinal Test Accuracy: {test_accuracy:.4f}')
