import torch
import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel



class Encoder(CNNBaseModel):

    def __init__(self,in_channels=3, encoder_classes=10, out_channels=3, init_weights=True):

        super(Encoder, self).__init__()
        # encoder
        self.conv_encoder1 = self._contracting_block(in_channels=in_channels, out_channels=64)
        self.max_pool_encoder1 = nn.MaxPool2d(kernel_size=2)
        self.conv_encoder2 = self._contracting_block(64, 128)
        self.max_pool_encoder2 = nn.MaxPool2d(kernel_size=2)
        self.conv_encoder3 = self._contracting_block(128, 256)
        self.max_pool_encoder3 = nn.MaxPool2d(kernel_size=2,padding=int(in_channels < 3))

        self.fc_layers_encode = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, encoder_classes)
        )

        self.fc_layers_decode = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(encoder_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 256 * 4 * 4),
            nn.ReLU(inplace=True)
        )

        self.transitional_block = torch.nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=int(in_channels > 1))
        )

        self.conv_decoder3 = self._expansive_block(512, 256, 128)
        self.conv_decoder2 = self._expansive_block(256, 128, 64)
        self.final_layer = self._final_block(128, 64, out_channels)
    
    def _contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        Building block of the contracting part
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block

    def _expansive_block(self, in_channels, mid_channels, out_channels, kernel_size=3):
        """
        Building block of the expansive part
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1,
                               output_padding=1)
        )
        return block

    def _final_block(self, in_channels, mid_channels, out_channels, kernel_size=3):
        """
        Final block of the UNet model
        """
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        # Encode
        encode_block1 = self.conv_encoder1(x)
        encode_pool1 = self.max_pool_encoder1(encode_block1)
        encode_block2 = self.conv_encoder2(encode_pool1)
        encode_pool2 = self.max_pool_encoder2(encode_block2)
        encode_block3 = self.conv_encoder3(encode_pool2)
        encode_pool3 = self.max_pool_encoder3(encode_block3)

        size_pool3 = encode_pool3.size()
        linear = encode_pool3.view(encode_pool3.size(0), -1)
        linear = self.fc_layers_encode(linear)
        linear = self.fc_layers_decode(linear)
        encode_pool3 = linear.view(size_pool3)

        # Transitional block
        middle_block = self.transitional_block(encode_pool3)

        # Decode
        decode_block3 = torch.cat((middle_block, encode_block3), 1)
        cat_layer2 = self.conv_decoder3(decode_block3)
        decode_block2 = torch.cat((cat_layer2, encode_block2), 1)
        cat_layer1 = self.conv_decoder2(decode_block2)
        decode_block1 = torch.cat((cat_layer1, encode_block1), 1)
        final_layer = self.final_layer(decode_block1)
        return final_layer
    
    def encode(self, x):
        # Encode
        encode_block1 = self.conv_encoder1(x)
        encode_pool1 = self.max_pool_encoder1(encode_block1)
        encode_block2 = self.conv_encoder2(encode_pool1)
        encode_pool2 = self.max_pool_encoder2(encode_block2)
        encode_block3 = self.conv_encoder3(encode_pool2)
        encode_pool3 = self.max_pool_encoder3(encode_block3)
        
        size_pool3 = encode_pool3.size()
        linear = encode_pool3.view(encode_pool3.size(0), -1)
        linear = self.fc_layers_encode(linear)

        return linear
