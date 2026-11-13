
import torch
import torch.nn as nn


class CRNN(nn.Module):

    def __init__(
        self,
        num_classes=118,
        hidden_size=128
    ):
        super(CRNN, self).__init__()

        # --------------------------------
        # 1. CNN: Visual Feature Extraction
        # --------------------------------

        self.cnn = nn.Sequential(

            # Input: (B, 1, 64, W)

            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Output: (B, 64, 32, W/2)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Output: (B, 128, 16, W/4)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1)
            ),

            # Output: (B, 256, 8, W/4)

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1)
            ),

            # Output: (B, 512, 4, W/4)
        )

        # --------------------------------
        # 2. BiLSTM: Sequence Learning
        # --------------------------------

        self.rnn = nn.LSTM(
            input_size=512 * 4,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )

        # --------------------------------
        # 3. Linear: Character Prediction
        # --------------------------------

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    def forward(self, images):

        # CNN feature extraction
        features = self.cnn(images)

        # (B, C, H, W) -> (B, W, C, H)
        features = features.permute(0, 3, 1, 2)

        # Combine channel and height dimensions
        batch_size, width, channels, height = features.size()

        features = features.reshape(
            batch_size,
            width,
            channels * height
        )

        # BiLSTM sequence processing
        sequence, _ = self.rnn(features)

        # Character class predictions
        output = self.classifier(sequence)

        return output
