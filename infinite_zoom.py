import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageOps
import numpy as np
import cv2
import os

#Configuration
DEVICE = "cuda"
MODEL_ID = "runwayml/stable-diffusion-inpainting"
PROMPT = "A colorful painting on a wall - of a painting on a wall, cinematic lighting, photorealistic, 8k"
NEGATIVE_PROMPT = "rainbow, psychedelic, cartoon, anime, painting, text, people"
NUM_FRAMES = 10
ZOOM_STEP = 0.90  #90% of new
OUTPUT_VIDEO = "infinite_zoom.mp4"
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 300
GUIDANCE_SCALE = 50
ZOOM_IN_EFFECT = True
OUTPUT_DIR = "frames"
FPS = 30
INTERPOLATION_STEPS = 30 #Lower is faster.


def dummy(images, **kwargs):
    return images, [False] * len(images)

def zoom_out_step(image, zoom_factor):
    """
    Creates a new image where the input 'image' is shrunk and placed in the center.
    Returns the new canvas and a mask (white borders, black center).
    """
    width, height = image.size
    
    #Shrink the image
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    
    shrunk_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    #Create background
    background = Image.new('RGB', (width, height), (0, 0, 0))
    paste_x = (width - new_width) // 2
    paste_y = (height - new_height) // 2
    background.paste(shrunk_image, (paste_x, paste_y))
    
    #Create mask
    mask = Image.new('L', (width, height), 255)
    #Black center (keep content)
    center_mask = Image.new('L', (new_width, new_height), 0)
    mask.paste(center_mask, (paste_x, paste_y))
    
    return background, mask

def write_video(frames, filename, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (WIDTH, HEIGHT))
    
    steps_per_keyframe = INTERPOLATION_STEPS 
    
    if ZOOM_IN_EFFECT:
        #Iterate backwards [N, N-1, ... 1]
        print("Rendering Zoom-In video...")
        for i in range(len(frames) - 1, 0, -1):
            img_wide = np.array(frames[i])
            img_wide = cv2.cvtColor(img_wide, cv2.COLOR_RGB2BGR)
            
            for step in range(steps_per_keyframe):
                t = step / steps_per_keyframe
                
                current_scale = 1.0 - (1.0 - ZOOM_STEP) * t
                
                crop_w = int(WIDTH * current_scale)
                crop_h = int(HEIGHT * current_scale)
                
                cx, cy = WIDTH // 2, HEIGHT // 2
                x1 = cx - crop_w // 2
                y1 = cy - crop_h // 2
                x2 = x1 + crop_w
                y2 = y1 + crop_h
                
                x2 = min(x2, WIDTH)
                y2 = min(y2, HEIGHT)
                x1 = max(x1, 0)
                y1 = max(y1, 0)

                crop = img_wide[y1:y2, x1:x2]
                
                if crop.size > 0:
                    frame = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
                    out.write(frame)
    else:
        #Iterate forwards [0, 1, ... N-1]
        print("Rendering Zoom-Out video...")
        for i in range(len(frames) - 1):
            img_next = np.array(frames[i+1])
            img_next = cv2.cvtColor(img_next, cv2.COLOR_RGB2BGR)

            for step in range(steps_per_keyframe):
                t = step / steps_per_keyframe
                
                current_scale = ZOOM_STEP + (1.0 - ZOOM_STEP) * t
                
                crop_w = int(WIDTH * current_scale)
                crop_h = int(HEIGHT * current_scale)
                
                cx, cy = WIDTH // 2, HEIGHT // 2
                x1 = cx - crop_w // 2
                y1 = cy - crop_h // 2
                x2 = x1 + crop_w
                y2 = y1 + crop_h
                
                x2 = min(x2, WIDTH)
                y2 = min(y2, HEIGHT)
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                
                crop = img_next[y1:y2, x1:x2]
                
                if crop.size > 0:
                    frame = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
                    out.write(frame)
            
    out.release()

def main():
    print(f"Loading model on {DEVICE}...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    pipe = pipe.to(DEVICE)
    pipe.safety_checker = dummy

    #Generate Initial Image
    print("Generating initial image...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_image = Image.new('RGB', (WIDTH, HEIGHT), (128, 128, 128))
    init_mask = Image.new('L', (WIDTH, HEIGHT), 255)
    
    current_image = pipe(
        prompt=PROMPT,
        image=init_image,
        mask_image=init_mask,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        negative_prompt=NEGATIVE_PROMPT
    ).images[0]
    
    current_image.save(os.path.join(OUTPUT_DIR, "frame_000.png"))
    frames = [current_image]
    
    #Zoom Loop
    for i in range(NUM_FRAMES):
        print(f"Generating frame {i+1}/{NUM_FRAMES}...")
        
        canvas, mask = zoom_out_step(current_image, ZOOM_STEP)
        
        #Inpaint borders
        result = pipe(
            prompt=PROMPT,
            image=canvas,
            mask_image=mask,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            negative_prompt=NEGATIVE_PROMPT
        ).images[0]
        
        current_image = result
        current_image.save(os.path.join(OUTPUT_DIR, f"frame_{i+1:03d}.png"))
        frames.append(current_image)

    #Create Video
    print("Rendering video...")
    write_video(frames, OUTPUT_VIDEO, fps=FPS)
    print("Done!")

if __name__ == "__main__":
    main()
