import os
import torch
import pandas as pd
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor, 
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings("ignore")

# ================= CONFIGURATION DU MEILLEUR SCORE (0.876) =================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
BASE_DIR = "custom_dataset"
OUTPUT_FILE = "submission_best.csv"

# CES PARAMÈTRES ONT PROUVÉ LEUR EFFICACITÉ
EPOCHS = 5               # Durée idéale : ni trop court (0.42), ni trop long (Collapse)
BATCH_SIZE = 4           
GRAD_ACCUM = 4           
LR = 5e-5                # Vitesse parfaite : assez lente pour stabiliser le Rank 128
LORA_RANK = 128          # Gros cerveau pour la précision
LORA_ALPHA = 256         

device = "cuda"

# ================= DATASET =================
class ClevrDataset(Dataset):
    def __init__(self, csv_file, img_dir, processor, mode="train"):
        self.data = pd.read_csv(csv_file)
        self.data.columns = self.data.columns.str.strip()
        
        # Renommage robuste
        rename_map = {}
        for col in self.data.columns:
            if col in ['file', 'filename']: rename_map[col] = 'image'
            if col in ['explanations', 'explanation_text']: rename_map[col] = 'explanation'
        if rename_map: self.data.rename(columns=rename_map, inplace=True)
        
        # Parsing explications
        if 'explanation' in self.data.columns:
            self.data['explanation'] = self.data['explanation'].apply(
                lambda x: eval(x)[0] if isinstance(x, str) and x.startswith("[") else x
            )

        self.img_dir = img_dir
        self.processor = processor
        self.mode = mode

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.img_dir, row['image'])
        try: image = Image.open(image_path).convert("RGB")
        except: image = Image.new('RGB', (224, 224), color='black')

        # Prompt qui a marché
        prompt = f"Question: {row['question']}\nFormat:\nAnswer: ...\nExplanation: ..."
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        
        if self.mode == "train":
            ans = row.get('answer', 'unknown')
            expl = row.get('explanation', 'none')
            messages.append({"role": "assistant", "content": [{"type": "text", "text": f"Answer: {ans}\nExplanation: {expl}"}]})

        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=(self.mode != "train"))
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_input], 
            images=image_inputs, 
            videos=video_inputs, 
            padding="max_length", 
            max_length=512, 
            return_tensors="pt"
        )
        
        # Fix Labels
        if self.mode == "train":
            inputs["labels"] = inputs["input_ids"]
        
        return {k: v.squeeze(0) for k, v in inputs.items()}

# Fix Collate (Garde grid_thw)
def collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        collated[key] = torch.stack([x[key] for x in batch])
    return collated

# ================= MAIN =================
if __name__ == "__main__":
    print(f">>> DÉMARRAGE CONFIGURATION GAGNANTE (LR={LR}, Rank={LORA_RANK}, Epochs={EPOCHS})")
    
    # 1. MODEL
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=1280*28*28)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", attn_implementation="eager", torch_dtype=torch.bfloat16
    )
    
    # 2. LORA
    peft_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 3. TRAINING
    train_ds = ClevrDataset(os.path.join(BASE_DIR, "train_labels.csv"), os.path.join(BASE_DIR, "train"), processor, "train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    num_training_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=num_training_steps)
    
    model.train()
    for epoch in range(EPOCHS):
        print(f"\n=== EPOCH {epoch+1}/{EPOCHS} ===")
        pbar = tqdm(train_loader)
        for i, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items() if v is not None}
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            if (i + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar.set_description(f"Loss: {outputs.loss.item():.4f}")
        
        # Sauvegarde
        model.save_pretrained(f"checkpoints_best/epoch_{epoch+1}")
        model.save_pretrained("fine_tuned_best")

    print(">>> DÉBUT INFÉRENCE...")
    torch.cuda.empty_cache()
    gc.collect()
    model.eval()
    
    test_ds = ClevrDataset(os.path.join(BASE_DIR, "test_non_labels.csv"), os.path.join(BASE_DIR, "test"), processor, "test")
    results = []
    
    for i in tqdm(range(len(test_ds))):
        try:
            row = test_ds.data.iloc[i]
            img_path = os.path.join(BASE_DIR, "test", row['image'])
            image = Image.open(img_path).convert("RGB")
            prompt = f"Question: {row['question']}\nFormat:\nAnswer: ...\nExplanation: ..."
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text_input], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
            
            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=128)
            output = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            
            ans, expl = "unknown", output
            clean = output.strip()
            if "Answer:" in clean:
                parts = clean.split("Answer:")[1]
                if "Explanation:" in parts:
                    ans = parts.split("Explanation:")[0].strip()
                    expl = parts.split("Explanation:")[1].strip()
                else: ans = parts.strip()
            
            results.append({"id": row['id'], "answer": ans, "explanation": expl})
        except:
            results.append({"id": row['id'], "answer": "err", "explanation": "err"})

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f">>> FINI. Fichier {OUTPUT_FILE} généré.")
