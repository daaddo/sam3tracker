# predict.py
from cog import BasePredictor, Input, Path
import torch
import numpy as np
from PIL import Image
from transformers import Sam3TrackerProcessor, Sam3TrackerModel
import os
import shutil
import time
import json
import logging
from typing import List

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

class Predictor(BasePredictor):
    def setup(self):
        """Carica il modello Tracker in memoria"""
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        log("Inizio Setup Tracker...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Using device: {self.device}")

        # 1. Carica Processor e Model specifici per il TRACKER
        # Nota: Usiamo Sam3TrackerProcessor e Sam3TrackerModel
        repo_id = "Davidinos/sam3data" # O il tuo path locale se preferisci
        
        log(f"Caricamento Tracker Processor da {repo_id}...")
        self.processor = Sam3TrackerProcessor.from_pretrained(repo_id)
        
        log(f"Caricamento Tracker Model da {repo_id}...")
        self.model = Sam3TrackerModel.from_pretrained(repo_id).to(self.device)
        
        log("Model Tracker loaded successfully.")

    def predict(
        self,
        image: Path = Input(description="Immagine da analizzare"),
        positive_points: str = Input(
            description="Lista JSON di punti positivi (da includere). Es: [[500, 300], [550, 320]]", 
            default="[]"
        ),
        negative_points: str = Input(
            description="Lista JSON di punti negativi (da escludere/sfondo). Es: [[100, 100]]", 
            default="[]"
        ),
    ) -> Path:
        """Esegue il raffinamento della maschera basato su punti"""
        log(f"--- Nuova Richiesta Tracker ---")
        
        # Gestione Output Dir
        output_dir = Path("/tmp/mask_result")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Caricamento Immagine
        log(f"Caricamento immagine: {image}")
        pil_image = Image.open(image).convert("RGB")
        width, height = pil_image.size
        log(f"Dimensioni: {width}x{height}")

        # 2. Parsing dei Punti (String -> List)
        try:
            pos_pts = json.loads(positive_points)
            neg_pts = json.loads(negative_points)
        except json.JSONDecodeError:
            raise ValueError("Errore nel formato dei punti. Usa formato JSON valido es: [[x,y], [x,y]]")

        # Uniamo punti e labels
        all_points = pos_pts + neg_pts
        all_labels = [1] * len(pos_pts) + [0] * len(neg_pts)

        if len(all_points) == 0:
            raise ValueError("Devi fornire almeno un punto positivo o negativo.")

        log(f"Punti totali: {len(all_points)} (Pos: {len(pos_pts)}, Neg: {len(neg_pts)})")

        # 3. Formattazione per SAM 3 (Nesting 4D richiesto)
        # Shape Points: (Batch, Num_Objects, Num_Points, 2) -> (1, 1, N, 2)
        # Shape Labels: (Batch, Num_Objects, Num_Points)    -> (1, 1, N)
        fmt_points = [[all_points]] 
        fmt_labels = [[all_labels]]

        # 4. Esecuzione Inferenza
        log("Esecuzione SAM3 Tracker Inference...")
        start_inf = time.time()
        
        inputs = self.processor(
            images=pil_image,
            input_points=fmt_points,
            input_labels=fmt_labels,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        log(f"Inferenza completata in {time.time() - start_inf:.2f}s")

        # 5. Selezione Maschera Migliore (Logica IoU Score)
        # outputs.iou_scores ha shape (Batch=1, Num_Obj=1, 3)
        scores = outputs.iou_scores
        best_mask_idx = torch.argmax(scores, dim=-1)
        best_idx_val = best_mask_idx[0, 0].item() # Estraiamo l'intero (0, 1 o 2)
        
        log(f"Selezionata maschera con indice {best_idx_val} (basato su IoU score)")

        # 6. Post-processing
        all_masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"], 
            mask_threshold=0.0, 
            binarize=True
        )[0] # Prende Batch 0 -> Shape: (Num_Obj=1, 3, H, W)

        # Seleziona l'Oggetto 0 e la variante Maschera vincente
        final_mask = all_masks[0, best_idx_val] # Shape (H, W)

        # 7. Salvataggio
        log("Salvataggio risultato...")
        # Convertiamo Tensor Bool -> Numpy Bool -> Uint8 (0-255)
        mask_np = final_mask.cpu().numpy()
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        
        mask_pil = Image.fromarray(mask_uint8, mode="L")
        output_path = output_dir / "refined_mask.png"
        mask_pil.save(output_path)
        
        log(f"Maschera salvata in {output_path}")
        return output_path