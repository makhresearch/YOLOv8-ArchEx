from ultralytics import YOLO

# Using if __name__ == '__main__': is a standard good practice in Python,
# which ensures that the code runs only when this file is executed directly.
if __name__ == '__main__':
    # --- Step 1: Load the custom architecture ---
    # We define the model using the YAML file you created.
    # The ultralytics library automatically finds this file in its standard path.
    print("Loading custom architecture from my-yolo.yaml...")
    model = YOLO('my-yolo.yaml')

    # --- Step 2: Load pre-trained weights (Transfer Learning) ---
    # This step is very important. We load the weights of a standard model (yolov8n.pt)
    # onto our new architecture. This allows common layers to have good initial
    # weights, helping the model to train much faster.
    print("Loading pre-trained weights from yolov8n.pt...")
    model.load('yolov8n.pt')

    # --- Step 3: Start the training process ---
    # Now, we train the model with the desired dataset.
    # For this example, we use coco128.yaml, which is a small sample dataset.
    # You can replace this with the path to your own dataset's YAML file.
    print("Starting training process...")
    results = model.train(
        data='coco128.yaml',       # Path to the dataset file
        epochs=5,                   # Number of training epochs (you can use 5 or 10 for a quick test)
        imgsz=640,                  # Input image size for the model
        batch=8,                    # Number of images processed in each batch
        name='yolo_custom_model_run'  # Name of the directory where results will be saved
    )

    # --- Step 4: Finish ---
    print("Training completed successfully!")
    print(f"Results and final weights are saved in the directory: '{results.save_dir}'")