import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from tqdm import tqdm
import os

from model import Generator, Discriminator, weights_init
from dataset import PhongDataset

import torchvision.models as models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Definicje globalne (potrzebne do normalizacji VGG)
VGG_WEIGHT = 0.6 
WARMUP_EPOCHS = 60
VGG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
VGG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

def setup_vgg_model(device):
    
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device)
    
    
    feature_layers = [4, 9, 16] 
    blocks = []
    
    current_layer = 0
    for i in feature_layers:
        block = nn.Sequential(*vgg[current_layer:i]).eval()
        for p in block.parameters():
            p.requires_grad = False  
        blocks.append(block)
        current_layer = i

    return nn.ModuleList(blocks).to(device)

def calculate_vgg_loss(vgg_model, fake_img, real_img):
    fake = nn.functional.interpolate(fake_img, size=(224, 224), mode='bilinear', align_corners=False)
    real = nn.functional.interpolate(real_img, size=(224, 224), mode='bilinear', align_corners=False)

    fake = (fake - VGG_MEAN) / VGG_STD
    real = (real - VGG_MEAN) / VGG_STD
    
    
    loss = 0.0
    for block in vgg_model:
        fake = block(fake)
        real = block(real)
        
        loss += torch.nn.functional.l1_loss(fake, real) 
        
    return loss



LEARNING_RATE_GEN = 0.0001    
LEARNING_RATE_DISC = 0.000005  
BATCH_SIZE = 32       
NUM_EPOCHS = 200      
L1_LAMBDA = 20.0   

LOAD_MODEL = True 
START_EPOCH =59

DISC_WARMUP_EPOCHS = 0  

dataset_dir = "/content/dataset"
checkpoint_dir = "checkpoints_C"
evaluation_dir = "evaluation_samples_C" #v5 uproszcozny D bardziej #v4 - uproszczony ORAZ zmiana loss masked na 4 zamiast 20 D (bo overfitting) #v3 - dynamic

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(evaluation_dir, exist_ok=True)



def train_cooldown_fn(gen, loader, opt_gen, l1_loss, epoch):
    loop = tqdm(loader, leave=True)
    
    for idx, (inputs, real_images) in enumerate(loop):
        inputs = inputs.to(DEVICE)
        real_images = real_images.to(DEVICE)
        
        
        fake_images = gen(None, inputs)
        
        loss_g_l1, raw_l1, _ = calculate_masked_loss(fake_images, real_images, lambda_val=1.0) # Lambda nie ma znaczenia bo jest jeden loss
        
        g_loss = loss_g_l1

        gen.zero_grad()
        g_loss.backward()
        opt_gen.step()

        loop.set_postfix(Mode="COOLDOWN", L1=f"{raw_l1.item():.5f}")

def calculate_masked_loss(fake, real, lambda_val):
    
    l1_diff = torch.abs(fake - real)
    
    
    mask = (real > -0.98).float()
    mask_pct = mask.mean().item() # Ile % obrazka to kula
    
     weights = 1.0 + (mask * 10)
    
    
    loss = (l1_diff * weights).mean() * lambda_val
    
    return loss, l1_diff.mean(), mask_pct

def save_checkpoint(model, optimizer, filename="checkpoint.pth"):
    print(f"=> Zapisywanie checkpointu do {filename}")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print(f"=> Wczytywanie checkpointu {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def get_loaders(root_dir, batch_size):
    # Podział: Train (2200), Val (200), Test (600)
    # Deterministyczny - manual_seed(42)
    
    dataset = PhongDataset(root_dir)
    
    total_len = len(dataset)
    test_len = 600             
    val_len = 200             
    train_len = total_len - test_len - val_len # (2200)

    generator = torch.Generator().manual_seed(42)
    
    train_ds, val_ds, test_ds = random_split(
        dataset, 
        [train_len, val_len, test_len], 
        generator=generator
    )

    print(f"Podział danych: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0) #pod cpu
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

def check_accuracy(val_loader, gen, disc, device, epoch, l1_loss_fn, bce_loss_fn, folder=evaluation_dir):
    gen.eval()
    disc.eval()
    

    total_raw_l1 = 0.0
    total_masked_l1 = 0.0
    total_g_loss = 0.0 
    total_d_loss = 0.0
    
    total_fake_score = 0.0 
    total_real_score = 0.0
    
    correct_real = 0
    correct_fake = 0
    total_samples = 0
    num_batches = 0

    print(f"\n=== RAPORT VALIDATION (EPOCH {epoch}) ===")
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.shape[0]
            
            fake_images = gen(None, inputs)
            
         
            # Real
            d_real = disc(targets, inputs)
            loss_d_real = bce_loss_fn(d_real, torch.ones_like(d_real)) # Bez label smoothing w eval
            
            # Fake
            d_fake = disc(fake_images, inputs)
            loss_d_fake = bce_loss_fn(d_fake, torch.zeros_like(d_fake))
            
            # Loss D
            d_loss = (loss_d_real * 1.5 + loss_d_fake * 0.5) /2
            total_d_loss += d_loss.item()
            
            
            total_real_score += d_real.mean().item()
            total_fake_score += d_fake.mean().item()
            
            
            correct_real += (d_real > 0.5).sum().item()
            correct_fake += (d_fake < 0.5).sum().item()
            total_samples += batch_size

            
            loss_g_gan = bce_loss_fn(d_fake, torch.ones_like(d_fake))
            
            masked_l1, raw_l1, _ = calculate_masked_loss(fake_images, targets, L1_LAMBDA)
            
            g_loss = loss_g_gan + masked_l1
            
            total_g_loss += g_loss.item()
            total_masked_l1 += masked_l1.item()
            total_raw_l1 += raw_l1.item()
            
            num_batches += 1

            if i == 0:
                if False: #wylaczenie

                    for j in range(min(16, len(targets))):

                        # Pojedyncze pary

                        real_img = targets[j:j+1]

                        fake_img = fake_images[j:j+1]

                       

                        # Pomiary lokalne

                        l1_val = torch.abs(fake_img - real_img).mean().item()

                        mask = (real_img > -0.98).float()

                        mask_p = mask.mean().item()

                       

                        weights = 1.0 + (mask * 19.0)

                        l1_weighted = (torch.abs(fake_img - real_img) * weights).mean().item() * L1_LAMBDA


                        sc_real = d_real[j].item()

                        sc_fake = d_fake[j].item()

                       

                        status = ""

                        if sc_fake > 0.5: status += "OSZUKANY! "

                       

                        print(f"{j:<3} | {sc_real:.4f}   | {sc_fake:.4f}   | {l1_val:.4f}   | {l1_weighted:.4f}   | {mask_p*100:.1f}%  | {status}") 

                debug_mask = (targets > -0.98).float()
                img_grid = torch.cat((targets[:16], fake_images[:16], debug_mask[:16]), dim=0)
                save_image(img_grid * 0.5 + 0.5, f"{folder}/epoch_{epoch}.png")

    avg_raw_l1 = total_raw_l1 / num_batches
    avg_masked_l1 = total_masked_l1 / num_batches
    avg_g_loss = total_g_loss / num_batches
    avg_d_loss = total_d_loss / num_batches
    
    avg_real_score = total_real_score / num_batches
    avg_fake_score = total_fake_score / num_batches
    
    acc_real = correct_real / total_samples
    acc_fake = correct_fake / total_samples

    print("-" * 75)
    print(f"LOSSES  => G: {avg_g_loss:.4f} | D: {avg_d_loss:.4f}")
    print(f"L1      => Raw: {avg_raw_l1:.4f} | Weighted: {avg_masked_l1:.4f}")
    print(f"SCORES  => Real(D): {avg_real_score:.4f} | Fake(D): {avg_fake_score:.4f}")
    print(f"ACC     => Real: {acc_real:.2%} | Fake: {acc_fake:.2%}")
    print("=" * 75)

    gen.train()
    disc.train()


ACC_TARGET_MIN = 0.65  
ACC_TARGET_MAX = 0.90  
MAX_D_REPEATS = 3      
MAX_G_REPEATS = 2      

CONSTANT_NOISE_STD = 0.01  

def train_dynamic_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, epoch, vgg_criterion):
    loop = tqdm(loader, leave=True)
    
    d_total_samples = 0
    d_real_correct = 0
    d_fake_correct = 0
    
    
    total_d_boosts = 0  
    total_g_boosts = 0  
    total_d_steps = 0   
    total_g_steps = 0   

    for idx, (inputs, real_images) in enumerate(loop):
        inputs = inputs.to(DEVICE)
        real_images = real_images.to(DEVICE)
        batch_size = inputs.shape[0]
        
        
        d_loops = 0
        
        while d_loops < MAX_D_REPEATS:
            with torch.no_grad():
                fake_images = gen(None, inputs)
            
            noise = torch.randn_like(real_images) * CONSTANT_NOISE_STD
            
            real_noisy = real_images + noise
            fake_noisy = fake_images.detach() + noise # Detach + Szum
            
            d_real = disc(real_noisy, inputs)
            loss_d_real = bce(d_real, torch.ones_like(d_real) * 0.9)
            
            d_fake = disc(fake_noisy, inputs)
            loss_d_fake = bce(d_fake, torch.zeros_like(d_fake))
            
            loss_d = (loss_d_real * 1.5 + loss_d_fake * 0.5) /2 #211
            
            acc_real = (d_real > 0.5).float().mean().item()
            acc_fake = (d_fake < 0.5).float().mean().item()
            current_acc = (acc_real + acc_fake) / 2
            
         
            
            is_unbalanced = abs(acc_real - acc_fake) > 0.4
            is_blind_to_real = acc_real < 0.45
            is_weak_overall = current_acc < ACC_TARGET_MAX
            
            # Decyzja o treningu D
            should_train_d = (is_weak_overall or is_blind_to_real or is_unbalanced)
            
            if current_acc > 0.98: should_train_d = False

            if should_train_d:
                disc.zero_grad()
                loss_d.backward()
                opt_disc.step()
                d_loops += 1
                total_d_steps += 1
            else:
                break 
            if current_acc > ACC_TARGET_MIN and not is_blind_to_real:
                break
        
        # Statystyki
        d_real_correct += (d_real > 0.5).sum().item()
        d_fake_correct += (d_fake < 0.5).sum().item()
        d_total_samples += batch_size
        
        if d_loops > 1: total_d_boosts += 1

    
        # Jeśli D jest za madry, G trenuje 2 razy
        g_repeats = MAX_G_REPEATS if current_acc >= ACC_TARGET_MAX else 1
        if g_repeats > 1: total_g_boosts += 1
        
        for i in range(g_repeats):
            fake_for_g = gen(None, inputs)
            noise_g = torch.randn_like(fake_for_g) * CONSTANT_NOISE_STD
            fake_g_noisy = fake_for_g + noise_g
            
            d_fake_g = disc(fake_g_noisy, inputs)
            
            loss_g_gan = bce(d_fake_g, torch.ones_like(d_fake_g))
            loss_g_l1, raw_l1, _ = calculate_masked_loss(fake_for_g, real_images, L1_LAMBDA)
            
            g_loss = loss_g_gan + loss_g_l1

            gen.zero_grad()
            g_loss.backward()
            opt_gen.step()
            total_g_steps += 1

        status_msg = "OK"
        if d_loops > 1: status_msg = f"D+{d_loops-1}"
        if g_repeats > 1: status_msg = f"G+{g_repeats-1}"

        loop.set_postfix(
            St=status_msg,
            D=f"{loss_d.item():.3f}", 
            G=f"{g_loss.item():.2f}", 
            L1=f"{raw_l1.item():.4f}", 
            L_W=f"{loss_g_l1.item():.2f}",
            AccR=f"{acc_real:.2f}", 
            AccF=f"{acc_fake:.2f}"
        )

    final_acc_real = d_real_correct / d_total_samples if d_total_samples > 0 else 0
    final_acc_fake = d_fake_correct / d_total_samples if d_total_samples > 0 else 0
    
    print(f"\n[Dynamic Stats] D_Boosts: {total_d_boosts} | G_Boosts: {total_g_boosts} || Total Steps: D={total_d_steps}, G={total_g_steps}")
    print(f"[Epoch Stats]   Acc Real: {final_acc_real:.2%} | Acc Fake: {final_acc_fake:.2%}")

def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, epoch, vgg_criterion):
    loop = tqdm(loader, leave=True)
    
    d_real_correct = 0
    d_fake_correct = 0
    d_total = 0

    for idx, (inputs, real_images) in enumerate(loop):
        inputs = inputs.to(DEVICE)
        real_images = real_images.to(DEVICE)
        batch_size = inputs.shape[0]
        
       #WARMUP
        if epoch < WARMUP_EPOCHS: 
            fake_images = gen(None, inputs)

            loss_g_l1, raw_l1, mask_pct = calculate_masked_loss(fake_images, real_images, L1_LAMBDA)
            
            # Tylko L1
            g_loss = loss_g_l1 

            gen.zero_grad()
            g_loss.backward()
            opt_gen.step()
            
            loop.set_postfix(Mode="WARMUP_L1", L1=f"{raw_l1:.4f}", TotalG=f"{g_loss.item():.2f}")
        
        else:
            freeze_gen = epoch < (START_EPOCH + DISC_WARMUP_EPOCHS)
            mode_name = "REHAB_DISC" if freeze_gen else "GAN_TRAIN"

            #  DYSKRYMINATOR 
            with torch.set_grad_enabled(not freeze_gen):
                fake_images = gen(None, inputs)
            
            # Real
            d_real = disc(real_images, inputs)
            loss_d_real = bce(d_real, torch.ones_like(d_real) * 0.9) # Label smoothing
            d_real_correct += (d_real > 0.5).sum().item()
            
            # Fake 
            d_fake = disc(fake_images.detach(), inputs)
            loss_d_fake = bce(d_fake, torch.zeros_like(d_fake))
            d_fake_correct += (d_fake < 0.5).sum().item()
            
            loss_d = (loss_d_real + loss_d_fake) / 2

            disc.zero_grad()
            loss_d.backward()
            opt_disc.step()

            #  GENERATOR
            if not freeze_gen:
                # Trenujemy G tylko w trybie GAN_TRAIN
                
                d_fake_for_gen = disc(fake_images, inputs)
                loss_g_gan = bce(d_fake_for_gen, torch.ones_like(d_fake_for_gen))

                loss_g_l1, raw_l1, mask_pct = calculate_masked_loss(fake_images, real_images, L1_LAMBDA)
                
                g_loss = loss_g_gan + loss_g_l1

                gen.zero_grad()
                g_loss.backward()
                opt_gen.step()
                
               
                log_g_loss = g_loss.item()
                log_l1 = raw_l1.item()
            else:
                with torch.no_grad():
                     _, log_l1, _ = calculate_masked_loss(fake_images, real_images, L1_LAMBDA)
                log_g_loss = 0.0

            # Logowanie
            d_total += batch_size
            acc_real = d_real_correct / d_total if d_total > 0 else 0
            acc_fake = d_fake_correct / d_total if d_total > 0 else 0
            
            loop.set_postfix(
                Mode=mode_name,
                D=f"{loss_d.item():.4f}", 
                G=f"{log_g_loss:.2f}", 
                L1=f"{log_l1:.4f}",
                AccR=f"{acc_real:.2f}", 
                AccF=f"{acc_fake:.2f}"
            )
def main():
    print(f"Urządzenie: {DEVICE}")
    print(f"Lambda L1: {L1_LAMBDA}")

    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    
    gen.apply(weights_init)
    disc.apply(weights_init)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_DISC, betas=(0.5, 0.999)) # disc uczy się 4 razy wolniej

    vgg_criterion = setup_vgg_model(DEVICE)

    bce = nn.BCELoss()
    l1_loss = nn.L1Loss()

    train_loader, val_loader = get_loaders(dataset_dir, BATCH_SIZE)

    if LOAD_MODEL:
        current_lr_gen = LEARNING_RATE_GEN
        current_lr_disc = LEARNING_RATE_DISC
        load_checkpoint(f"{checkpoint_dir}/gen_epoch_{START_EPOCH}.pth", gen, opt_gen, current_lr_gen)
        load_checkpoint(f"{checkpoint_dir}/disc_epoch_{START_EPOCH}.pth", disc, opt_disc, current_lr_disc)
        print(f"Wznowiono trening od epoki {START_EPOCH}. Nowe LR: G={current_lr_gen}, D={current_lr_disc}")

    #opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0.0, 0.999))

    for epoch in range(START_EPOCH + 1, START_EPOCH + 1 + NUM_EPOCHS):
        print(f"Epoch [{epoch}/{START_EPOCH + NUM_EPOCHS}]")
        #train_fn(disc, gen, train_loader, opt_disc, opt_gen, l1_loss, bce, epoch, vgg_criterion)
        #train_cooldown_fn(gen, train_loader, opt_gen, l1_loss, epoch)
        
        train_dynamic_fn(disc, gen, train_loader, opt_disc, opt_gen, l1_loss, bce, epoch, vgg_criterion)
        check_accuracy(val_loader, gen, disc, DEVICE, epoch, l1_loss,bce)

        save_checkpoint(gen, opt_gen, filename=f"{checkpoint_dir}/gen_epoch_{epoch}.pth")
        save_checkpoint(disc, opt_disc, filename=f"{checkpoint_dir}/disc_epoch_{epoch}.pth")

if __name__ == "__main__":
    main()

