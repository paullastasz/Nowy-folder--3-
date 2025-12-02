import torch
import torch.nn as nn


PARAM_DIM = 10          # Wymiar parametrów IN
IMG_CHANNELS = 3        # RGB
IMG_SIZE = 128          # 128x128

class Generator(nn.Module):
   
    def __init__(self):
        super(Generator, self).__init__()
        
        self.input_dim = PARAM_DIM

        self.initial_layer = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            
            nn.Linear(512, 512 * 4 * 4), 
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True)
        )
        
        self.model = nn.Sequential(
            
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 64x64 -> 128x128 (Output)
            nn.ConvTranspose2d(32, IMG_CHANNELS, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh() # Wyjście [-1, 1]
        )

    def forward(self, noise, labels):
        
        x = self.initial_layer(labels)
        
        x = x.view(-1, 512, 4, 4)
        img = self.model(x)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Sequential(
            nn.Linear(PARAM_DIM, 128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.image_processing = nn.Sequential(
            # Input: 3 x 128 x 128
            
            nn.Conv2d(IMG_CHANNELS, 16, kernel_size=4, stride=2, padding=1, bias=False), # Start od 16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False), # Max 32
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), # Max 64
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), # Max 128
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Obrazek: 128 kanałów * 8 * 8 pikseli = 8192
        self.flatten_size = 128 * 8 * 8
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size + 128, 256), 
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Dropout(0.5), 
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        features = self.image_processing(img)     
        features_flat = features.view(features.size(0), -1) 

        label_emb = self.label_embedding(labels)
        concat_input = torch.cat((features_flat, label_emb), dim=1) 

        validity = self.classifier(concat_input)
        return validity

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# test
if __name__ == "__main__":
    batch_size = 5
    noise = torch.randn(batch_size, LATENT_DIM)
    params = torch.randn(batch_size, PARAM_DIM)
    
    G = Generator()
    fake_imgs = G(noise, params)
    print(f"Generator output: {fake_imgs.shape}") #  [5, 3, 128, 128]

    D = Discriminator()
    validity = D(fake_imgs, params)
    print(f"Discriminator output: {validity.shape}") #  [5, 1]