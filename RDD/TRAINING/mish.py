# train_mish.py
import os
import sys
from ultralytics import YOLO
from ultralytics.nn import tasks

# Ensure the project directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import the custom activation definition
import custom_activations

# --- Manually Register Mish Activation ---
# This adds your custom Mish class to the scope the YOLO parser uses.
tasks.Mish = custom_activations.Mish
print("INFO: Manually registered 'Mish' activation with Ultralytics.")

def main():
    """ Main training function """
    print("--- Starting YOLOv8s MISH Training ---")

    # --- Configuration ---
    model_config_yaml = 'yolov8s-mish.yaml'
    dataset_config_yaml = 'rdd_dataset.yaml' # IMPORTANT: Change to your dataset's YAML file
    pretrained_weights = 'yolov8s.pt'
    
    # --- Initialize and Train ---
    try:
        model = YOLO(model_config_yaml).load(pretrained_weights)
        print("INFO: Model with Mish activation initialized successfully.")

        results = model.train(
            data=dataset_config_yaml,
            epochs=100,
            batch=16, # Adjust based on your GPU
            imgsz=640,
            name='yolov8s_mish_rdd_run'
        )
        
    except Exception as e:
        print(f"\nFATAL ERROR during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
