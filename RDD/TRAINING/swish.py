# train_swish_default.py
from ultralytics import YOLO

def main():
    """ Main training function for the default YOLOv8s model (uses SiLU/Swish) """
    print("--- Starting YOLOv8s SWISH (Default SiLU) Training ---")

    # --- Configuration ---
    # No custom model YAML needed, we load the standard architecture
    model_name = 'yolov8s.pt'
    dataset_config_yaml = 'rdd_dataset.yaml' # IMPORTANT: Change to your dataset's YAML file

    # --- Initialize and Train ---
    try:
        model = YOLO(model_name)
        
        # Inspect the activation function to confirm
        try:
            activation_name = model.model[0].act.__class__.__name__
            print(f"INFO: Confirmed default activation function is '{activation_name}'.")
        except Exception:
            print("INFO: Could not automatically inspect activation function, but it should be SiLU.")

        results = model.train(
            data=dataset_config_yaml,
            epochs=100,
            batch=16, # Adjust based on your GPU
            imgsz=640,
            name='yolov8s_swish_rdd_run'
        )
        
    except Exception as e:
        print(f"\nFATAL ERROR during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
