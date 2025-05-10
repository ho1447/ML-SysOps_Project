import torch.nn as nn

class CommandClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=35):
        super(CommandClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
