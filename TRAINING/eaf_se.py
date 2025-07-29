# train_eaf_se.py
import os
import sys
from ultralytics import YOLO
from ultralytics.nn import tasks

# Ensure the project directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import custom_eaf_se

# --- Manually Register Custom Modules ---
tasks.ConvEAF_SE = custom_eaf_se.ConvEAF_SE
tasks.C2fEAF_SE = custom_eaf_se.C2fEAF_SE
tasks.BottleneckEAF_SE = custom_eaf_se.BottleneckEAF_SE
print("INFO: Manually registered custom EAF+SE modules with Ultralytics.")

def main():
    """ Main training function for EAF+SE model """
    print("--- Starting YOLOv8s EAF+SE Training ---")

    # --- Configuration ---
    model_config_yaml = 'yolov8s_eaf_se.yaml'
    dataset_config_yaml = 'rdd_dataset.yaml' # IMPORTANT: Change to your dataset's YAML file
    pretrained_weights = 'yolov8s.pt'
    
    # --- Initialize and Train ---
    try:
        model = YOLO(model_config_yaml).load(pretrained_weights)
        print("INFO: Model with EAF+SE modules initialized successfully.")

        # EAF requires very careful hyperparameter tuning
        results = model.train(
            data=dataset_config_yaml,
            epochs=50,
            batch=4,       # Start very small
            imgsz=640,
            lr0=1e-5,      # CRITICAL: Start with a very low learning rate
            optimizer='AdamW',
            name='yolov8s_eaf_se_rdd_run'
        )
        
    except Exception as e:
        print(f"\nFATAL ERROR during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
