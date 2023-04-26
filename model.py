import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummaryX import summary

def activation_layer(activation: str='relu', alpha: float=0.1, inplace: bool=True):
    """ Activation layer wrapper for LeakyReLU and ReLU activation functions

    Args:
        activation: str, activation function name (default: 'relu')
        alpha: float (LeakyReLU activation function parameter)

    Returns:
        torch.Tensor: activation layer
    """
    if activation == 'relu':
        return nn.ReLU(inplace=inplace)
    
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=alpha, inplace=inplace)

class ConvBlock(nn.Module):
    """ Convolutional block with batch normalization
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_conv=True, stride=1, dropout=0.2, activation='leaky_relu'):
        super(ResidualBlock, self).__init__()
        self.convb1 = ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.act1 = activation_layer(activation)

        self.convb2 = ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(p=dropout)
        
        self.shortcut = None
        if skip_conv:
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        self.act2 = activation_layer(activation)
        
    def forward(self, x):
        skip = x
        
        out = self.act1(self.convb1(x))
        out = self.convb2(out)

        if self.shortcut is not None:
            out += self.shortcut(skip)

        out = self.act2(out)
        out = self.dropout(out)
        
        return out

class HandwrittenMongolianModel(nn.Module):
    """ Handwriting recognition network for CTC loss"""
    def __init__(self, num_chars: int, activation: str='leaky_relu', dropout: float=0.2):
        super(HandwrittenMongolianModel, self).__init__()

        self.rb1 = ResidualBlock(1, 16, skip_conv = True, stride=1, activation=activation, dropout=dropout)
        self.rb2 = ResidualBlock(16, 16, skip_conv = True, stride=2, activation=activation, dropout=dropout)
        self.rb3 = ResidualBlock(16, 16, skip_conv = False, stride=1, activation=activation, dropout=dropout)

        self.rb4 = ResidualBlock(16, 32, skip_conv = True, stride=2, activation=activation, dropout=dropout)
        self.rb5 = ResidualBlock(32, 32, skip_conv = False, stride=1, activation=activation, dropout=dropout)

        self.rb6 = ResidualBlock(32, 64, skip_conv = True, stride=2, activation=activation, dropout=dropout)
        self.rb7 = ResidualBlock(64, 64, skip_conv = True, stride=1, activation=activation, dropout=dropout)

        self.rb8 = ResidualBlock(64, 64, skip_conv = False, stride=1, activation=activation, dropout=dropout)
        self.rb9 = ResidualBlock(64, 64, skip_conv = False, stride=1, activation=activation, dropout=dropout)

        self.lstm = nn.LSTM(64, 128, bidirectional=True, num_layers=1, batch_first=True)
        self.lstm_dropout = nn.Dropout(p=dropout)

        self.output = nn.Linear(256, num_chars + 1)

    def forward(self, images: torch.Tensor, targets=None) -> torch.Tensor:
        # normalize images between 0 and 1
        images_flaot = images / 255.0

        # transpose image to channel first
        # images_flaot = images_flaot.permute(0, 3, 1, 2)

        bs, _, _, _ = images.size()

        # apply convolutions
        x = self.rb1(images_flaot)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.rb6(x)
        x = self.rb7(x)
        x = self.rb8(x)
        x = self.rb9(x)

        x = x.reshape(x.size(0), -1, x.size(1))

        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)

        x = self.output(x)
        # print(x.size())
        x = x.permute(1, 0, 2)
        # print(x.size())

        if targets is not None:
            log_probs = F.log_softmax(x, 2)
            # print(log_probs.size())
            input_lengths = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            target_lengths = torch.count_nonzero(targets, dim=1)
            # print(log_probs)
            # print(targets)
            # print(target_lengths)
            # remove zeros from targets each row has different length of zeros
            targets = [target[target != 0] for target in targets]
            # flatten targets
            targets = torch.cat(targets)
            loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')(
                log_probs, targets.cpu(), input_lengths.cpu(), target_lengths.cpu()
            )
            return x, loss

        return x, None

if __name__ == "__main__":
    model = HandwrittenMongolianModel(37)
    dummy_input = torch.randn(1, 1, 48, 300)
    summary(model, dummy_input)
    # save model to onnx
    # torch.onnx.export(model, dummy_input, 'model.onnx', input_names=["word image"], output_names=["predictions"])
    # forward pass through the model
    outputs, _ = model(dummy_input)
    print(f"Output shape: {outputs.shape}")