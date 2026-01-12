# Infinite Zoom Generator

This project generates stunning infinite zoom videos using Stable Diffusion Inpainting. By iteratively zooming out (or in) and filling the new canvas space with AI-generated content, it creates a seamless tunnel-like video effect.

## Features

- **Stable Diffusion Inpainting**: Uses the `runwayml/stable-diffusion-inpainting` model for high-quality image generation.
- **Customizable Prompts**: Define your own prompts and negative prompts to control the artistic style and content.
- **Adjustable Parameters**: Fine-tune the number of frames, zoom step size, inference steps, and guidance scale.
- **Video Generation**: Automatically compiles the generated frames into a smooth MP4 video using OpenCV.
- **Interpolation**: Smooths the transition between keyframes for a fluid video experience.
- **Direction Control**: Toggle between Zoom-In and Zoom-Out effects.

## Requirements

To run this script, you need Python installed along with the following libraries:

- `torch` (with CUDA support recommended for performance)
- `diffusers`
- `Pillow`
- `numpy`
- `opencv-python`

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Marcel-H1/infinite-zoom.git
    cd infinite-zoom
    ```

2.  Install the dependencies:
    ```bash
    pip install torch diffusers Pillow numpy opencv-python transformers accelerate
    ```

## Usage

1.  Open `infinite_zoom.py` and modify the **Configuration** section at the top of the file to match your preferences:

    ```python
    # Configuration
    PROMPT = "A colorful painting on a wall - of a painting on a wall, cinematic lighting, photorealistic, 8k"
    NUM_FRAMES = 10         # Number of keyframes to generate
    ZOOM_STEP = 0.90        # Zoom factor (0.90 means 90% of the previous size)
    ZOOM_IN_EFFECT = True   # Set to False for a Zoom-Out video
    # ... other settings
    ```

2.  Run the script:
    ```bash
    python infinite_zoom.py
    ```

3.  The script will:
    - Generate the initial image.
    - Iteratively zoom and inpaint to create subsequent frames (saved in the `frames/` directory).
    - Render the final video as `infinite_zoom.mp4`.


## Output

- **Frames**: Individual generated images are stored in the `frames` folder.
- **Video**: The final animation is saved as `infinite_zoom.mp4`.