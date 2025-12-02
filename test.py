import torch
import os
import shutil
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

#ustaw dobry model!!
from model_3 import Generator
from dataset import PhongDataset

# --- KONFIGURACJA ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_DIR = "/content/dataset"  
CHECKPOINT_PATH = "checkpoints_3/gen_epoch_270.pth" 
OUTPUT_ROOT = "test_results"      # Gdzie zapisać wyniki
MODEL_PREFIX = "model_3"


def get_test_loader_standalone(root_dir):
    #podział danych identycznie jak w train.py
    
    dataset = PhongDataset(root_dir)
    
    # Te same długości co w train.py
    total_len = len(dataset)
    test_len = 600             
    val_len = 200             
    train_len = total_len - test_len - val_len # 2200

    # KLUCZOWE: Ten sam seed co w train.py
    generator = torch.Generator().manual_seed(42)
    
    # Wykonujemy podział
    _, _, test_ds = random_split(
        dataset, 
        [train_len, val_len, test_len], 
        generator=generator
    )

    print(f"Odtworzono zbiór testowy: {len(test_ds)} próbek.")
    
    # Batch size 1 dla łatwiejszego kopiowania plików
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    
    return test_loader

def setup_directories(model_name):
    base_path = os.path.join(OUTPUT_ROOT, model_name)
    target_dir = os.path.join(base_path, "target")
    pred_dir = os.path.join(base_path, "pred")
    
    if os.path.exists(base_path):
        print(f"UWAGA! Folder {base_path} już istnieje. Nadpisywanie...")
    
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    
    return target_dir, pred_dir

def load_generator(path):
    print(f"=> Wczytywanie modelu z: {path}")
    gen = Generator().to(DEVICE)
    
    checkpoint = torch.load(path, map_location=DEVICE)
    
    # Obsługa zapisu state_dict vs pełny checkpoint
    if "state_dict" in checkpoint:
        gen.load_state_dict(checkpoint["state_dict"])
    else:
        gen.load_state_dict(checkpoint)
        
    gen.eval() # Ważne! Wyłącza dropout i batchnorm statystyki
    return gen

def run_test():
    # Parsowanie nazwy folderu z pliku checkpointu (np. gen_epoch_204.pth -> 204)
    filename = os.path.basename(CHECKPOINT_PATH)
    epoch_number = filename.replace(".pth", "").split("_")[-1]
    
    # model_X_204
    folder_name = f"{MODEL_PREFIX}_{epoch_number}"
    
    target_dir, pred_dir = setup_directories(folder_name)
    
    test_loader = get_test_loader_standalone(DATASET_DIR)
    
    gen = load_generator(CHECKPOINT_PATH)
    
    subset = test_loader.dataset
    original_dataset = subset.dataset
    indices = subset.indices

    print(f"\nGenerowanie i kopiowanie do: {target_dir} i {pred_dir}... z checkpoint: {CHECKPOINT_PATH}")
    
    total_l1_loss = 0.0
    count = 0

    with torch.no_grad():
        for i, (inputs, real_imgs) in enumerate(tqdm(test_loader)):            
            inputs = inputs.to(DEVICE)
            real_imgs = real_imgs.to(DEVICE)

            # Generacja
            fake_img = gen(None, inputs)

            current_l1 = torch.abs(fake_img - real_imgs).mean().item()
            total_l1_loss += current_l1
            count += 1
            
            # Identyfikacja pliku
            original_idx = indices[i]
            json_filename = original_dataset.files[original_idx]
            
            # Ścieżki źródłowe
            src_json_path = os.path.join(original_dataset.root_dir, json_filename)
            
            # Odczyt nazwy obrazka z JSONa
            with open(src_json_path, 'r') as f:
                meta = json.load(f)
            img_filename = meta["file_name"]
            src_img_path = os.path.join(original_dataset.root_dir, img_filename)
            
            # ZAPIS
            
            # PRED
            save_image(fake_img * 0.5 + 0.5, os.path.join(pred_dir, img_filename))
            
            # TARGET (Kopia oryginału)
            shutil.copy(src_img_path, os.path.join(target_dir, img_filename))
            
            # JSON (Kopia parametrów)
            shutil.copy(src_json_path, os.path.join(target_dir, json_filename))

    avg_l1 = total_l1_loss / count
    
    print("\n" + "="*50)
    print(f" Przetworzono obrazów: {count}")
    print(f" Średni L1 Loss: {avg_l1:.6f}")
    print(f"Wyniki w folderze: {os.path.join(OUTPUT_ROOT, folder_name)}")

if __name__ == "__main__":
    run_test()