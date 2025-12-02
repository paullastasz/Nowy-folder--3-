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
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 64x64 -> 128x128
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, IMG_CHANNELS, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.Tanh() # Wyjście [-1, 1]
        )

    def forward(self, noise, labels):
        
        x = self.initial_layer(labels)
        
        # Formowanie obrazka
        x = x.view(-1, 512, 4, 4)
        img = self.model(x)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        
        self.label_embedding = nn.Sequential(
            nn.Linear(PARAM_DIM, 512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.image_processing = nn.Sequential(
            nn.Conv2d(IMG_CHANNELS, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Wejście: 8192 (Obraz) + 512 (Parametry) = 8704
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4 + 512, 512), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
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