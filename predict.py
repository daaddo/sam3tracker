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

        # Repository HuggingFace o Path Locale
        repo_id = "Davidinos/sam3data" 
        
        log(f"Caricamento Tracker Processor da {repo_id}...")
        self.processor = Sam3TrackerProcessor.from_pretrained(repo_id)
        
        log(f"Caricamento Tracker Model da {repo_id}...")
        self.model = Sam3TrackerModel.from_pretrained(repo_id).to(self.device)
        
        log("Model Tracker loaded successfully.")

    def predict(
        self,
        image: Path = Input(description="Immagine da analizzare"),
        
        # ESEMPIO INPUT: [[[15,14],[16,17]], [[11,10]]]
        # Lista esterna = Oggetti. Lista interna = Punti per quell'oggetto.
        objects_points: str = Input(
            description="JSON Lista di Liste di punti. Ogni lista interna è un oggetto. Es: [[[x1,y1],[x2,y2]], [[x3,y3]]]", 
            default="[[[500, 300]]]"
        ),
        
        # ESEMPIO LABELS: [[1, 0], [1]]
        # Deve corrispondere alla struttura dei punti. 1=Positivo, 0=Negativo
        objects_labels: str = Input(
            description="JSON Lista di Liste di labels (1=Pos, 0=Neg). Deve specchiare la struttura di objects_points.", 
            default="[[1]]"
        ),
    ) -> List[Path]:
        """
        Esegue il raffinamento per MULTIPLI oggetti contemporaneamente sulla stessa immagine.
        Restituisce una lista di percorsi alle maschere generate.
        """
        log(f"--- Nuova Richiesta Multi-Oggetto ---")
        
        # 1. Gestione Cartella Output
        output_dir = Path("/tmp/mask_results")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 2. Caricamento Immagine
        log(f"Caricamento immagine: {image}")
        pil_image = Image.open(image).convert("RGB")
        log(f"Dimensioni: {pil_image.size}")

        # 3. Parsing JSON Inputs
        try:
            # points_data sarà: List[List[List[int]]] -> [ [ [x,y], [x,y] ], [ [x,y] ] ]
            points_data = json.loads(objects_points)
            # labels_data sarà: List[List[int]]       -> [ [1, 0], [1] ]
            labels_data = json.loads(objects_labels)
        except json.JSONDecodeError:
            raise ValueError("Errore JSON nei punti o labels. Controlla la formattazione.")

        # Validazione base
        if len(points_data) != len(labels_data):
            raise ValueError(f"Mismatch: Hai passato punti per {len(points_data)} oggetti ma labels per {len(labels_data)} oggetti.")
        
        num_objects = len(points_data)
        log(f"Rilevati {num_objects} oggetti da processare.")

        # 4. Formattazione per SAM 3 (Nesting 4D)
        # La struttura richiesta è: [ Batch_Images [ List_Objects [ List_Points [x,y] ] ] ]
        # Dato che abbiamo 1 sola immagine, avvolgiamo tutto in una lista esterna.
        fmt_points = [points_data] 
        fmt_labels = [labels_data]

        # 5. Inferenza
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

        # 6. Post-processing
        # Restituisce: [Batch_Size, Num_Objects, 3, H, W] -> Qui Batch=1
        all_masks_batch = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"], 
            mask_threshold=0.0, 
            binarize=True
        )
        
        # Prendiamo la prima (e unica) immagine del batch
        # shape risultante: (Num_Objects, 3, H, W)
        object_masks = all_masks_batch[0] 
        
        # Recuperiamo gli scores: shape (1, Num_Objects, 3)
        iou_scores = outputs.iou_scores[0] # Ora shape (Num_Objects, 3)

        output_paths = []

        # 7. Ciclo su ogni oggetto per scegliere la maschera migliore e salvarla
        log(f"Salvataggio di {num_objects} maschere...")
        
        for i in range(num_objects):
            # A. Trova l'indice (0,1,2) migliore per l'oggetto corrente 'i'
            scores_obj = iou_scores[i]            # shape (3,)
            best_idx = torch.argmax(scores_obj).item()
            
            # B. Estrai la maschera vincente
            # object_masks ha shape (Num_Objects, 3, H, W)
            final_mask = object_masks[i, best_idx] # shape (H, W)

            # C. Salvataggio
            mask_np = final_mask.cpu().numpy()
            mask_uint8 = (mask_np * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_uint8, mode="L")
            
            filename = f"mask_obj_{i:02d}.png"
            file_path = output_dir / filename
            mask_pil.save(file_path)
            
            output_paths.append(file_path)
            log(f" -> Oggetto {i}: salvato {filename} (Score idx: {best_idx})")

        return output_paths