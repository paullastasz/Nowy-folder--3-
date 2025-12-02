import torch
import torch.nn as nn


PARAM_DIM = 10
IMG_CHANNELS = 3
IMG_SIZE = 128

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.input_dim = PARAM_DIM
        
        # Liniowa transformacja do 4x4
        # do 512
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        #Upsample (x2) -> CoordConv -> Conv2d -> BN -> LeakyReLU
        
        self.block1 = self._make_gen_block(512, 256) # 4 -> 8
        self.block2 = self._make_gen_block(256, 128) # 8 -> 16
        self.block3 = self._make_gen_block(128, 64)  # 16 -> 32
        self.block4 = self._make_gen_block(64, 32)   # 32 -> 64
        
        self.last_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            
            nn.Conv2d(32, IMG_CHANNELS, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def _make_gen_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), 
            
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, noise, labels):
       
        x = self.fc(labels)
        x = x.view(-1, 512, 4, 4) # [Batch, 512, 4, 4]
        
        x = self.block1(x) # 8x8
        x = self.block2(x) # 16x16
        x = self.block3(x) # 32x32
        x = self.block4(x) # 64x64
        
        img = self.last_block(x) # 128x128
        
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
            
            # Layer 1: 128 -> 64
            nn.Conv2d(IMG_CHANNELS, 16, kernel_size=4, stride=2, padding=1, bias=False), # Start od 16
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 64 -> 32
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False), # Max 32
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 32 -> 16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False), # Max 64
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 16 -> 8
            # Tu kończymy (nie schodzimy do 4x4). Obrazek 8x8 ma wystarczająco mało detali.
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False), # Max 128
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        
        self.flatten_size = 128 * 8 * 8
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size + 128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # DROPOUT 0.5 - agresywne zapobieganie pamieci
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