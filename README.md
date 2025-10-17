# YOLOv8-ArchEx: A Framework for Extending YOLOv8 Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is a proof-of-concept for customizing and extending the YOLOv8 neural network architecture. It demonstrates a complete workflow for replacing a standard network module with a custom-built one and successfully training the modified model.

This provides a solid foundation for task-specific adaptations, such as improving small object detection for applications like license plate recognition.

This project was developed under the supervision of **Dr. Mirhasani**.

---

## Key Features

-   **Modular Customization:** A clear example of how to add custom modules (`MyCustomConv`) in a separate, organized file (`src/custom_modules.py`).
-   **Architecture Configuration:** A custom YAML file (`src/my-yolo.yaml`) that integrates the new module into the YOLOv8 backbone.
-   **Core Library Patching:** Detailed instructions on how to patch the `ultralytics` core (`tasks.py`) to make it aware of the new custom components.
-   **Complete Documentation:** A full technical report is available in the `docs` directory, detailing the project's philosophy, challenges, and solutions.

## Project Structure

```
YOLOv8-ArchEx/
├── docs/
│   └── Customizable-YOLOv8-Technical-Report.pdf  # Full technical report
├── src/
│   ├── custom_modules.py                         # Custom PyTorch module
│   ├── my-yolo.yaml                              # Custom model architecture
│   └── train_custom.py                           # Training script
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt                              # Python dependencies
```

---

## Installation and Setup Guide

This repository does not contain the full `ultralytics` library. Instead, it provides the necessary components to **patch** an official installation. Follow these steps carefully.

### Step 1: Clone and Install the Base `ultralytics` Repository

First, you need the official YOLOv8 codebase.

```bash
# 1. Clone the official repository
git clone https://github.com/ultralytics/ultralytics
cd ultralytics

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# 4. Install the library in editable mode
pip install -e .
```

### Step 2: Apply the Custom Patches

Now, copy the custom files from this repository (`YOLOv8-ArchEx`) into the `ultralytics` folder you just cloned.

1.  **Copy Custom Module:**
    Copy `YOLOv8-ArchEx/src/custom_modules.py` to `ultralytics/ultralytics/nn/`.

2.  **Copy Custom Architecture:**
    Copy `YOLOv8-ArchEx/src/my-yolo.yaml` to `ultralytics/ultralytics/cfg/models/v8/`.

3.  **Patch the Core Model Parser (`tasks.py`):**
    This is the most critical step. Open `ultralytics/ultralytics/nn/tasks.py` and make the following two changes:
    *   **Import the custom module** at the top of the file:
        ```python
        from .custom_modules import MyCustomConv
        ```
    *   **Register the module.** Inside the `parse_model` function, find the `base_modules` frozenset and add `MyCustomConv` to the list (e.g., right after `Conv`):
        ```python
        base_modules = frozenset(
            {
                Classify,
                MyCustomConv,  # <-- Add this line
                Conv,
                # ... rest of the modules
            }
        )
        ```

### Step 3: Run the Custom Training

1.  **Copy Training Script:**
    Copy `YOLOv8-ArchEx/src/train_custom.py` into the **root** of the `ultralytics` directory.

2.  **Execute the training:**
    Make sure your virtual environment is still active, and run the script.
    ```bash
    python train_custom.py
    ```

You should now see the training process start, and the model summary will list `MyCustomConv` as one of the layers.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
