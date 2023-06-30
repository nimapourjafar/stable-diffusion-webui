import os
import uuid
import glob
import requests
import json
from pprint import pprint
import base64
from random import randint, choice
from io import BytesIO
from PIL import Image
import time
from moviepy.editor import ImageSequenceClip
import subprocess

script_directory = os.path.dirname(os.path.abspath(__file__))
fps = int(input("Enter FPS: "))

checkpoints = ["realistic"]
denoisingStrengths = [0.55,0.65,0.75,0.85]


promptDictionary = {
    "realistic": ["outside at beach, oceanic background, vibrant background, blue sky, sun is out, day-time (RAW photo:1.5), busty, women standing wearing white dress, cleavage, panties, juicy legs, masterpiece, sharp, best quality, (high detailed skin:1.2), dslr, soft lighting,  Depth of (Field:1.1)"],
    "disneyPixarCartoon": [""],
    "aniflatmix": ["masterpiece, sharo, best quality, city pop, night, neon light, busty, vector illustration, jacket, cleavage, light smile"]
}

badPromptDictionary = {
    "realistic": ["CGI, 3d, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"],
    "disneyPixarCartoon": [""],
    "aniflatmix": ["EasyNegative, badhandv4, monochrome, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"]
}

def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Generate the output filename pattern
    output_pattern = os.path.join(output_folder, 'frame-%04d.jpg')

    # Run FFmpeg command to extract frames
    ffmpeg_cmd = [
        'ffmpeg',
        '-i',
        video_path,
        '-vf',
        'fps={}'.format(fps),
        output_pattern
    ]
    subprocess.run(ffmpeg_cmd)

def get_video_length(video_path):
    # Run FFprobe command to get video information
    ffprobe_cmd = [
        'ffprobe',
        '-v',
        'error',
        '-show_entries',
        'format=duration',
        '-of',
        'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
    duration = float(result.stdout)

    return duration

def get_image_paths(folder):
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted([(path, get_image_dimensions(path)) for path in files])


def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size

def make_init_image(prompt, bad_prompt, denoising_strength, width, height, image_path):

    url = "http://localhost:7860/sdapi/v1/img2img"

    with open(image_path, "rb") as f:
        current_image = base64.b64encode(f.read()).decode("utf-8")


    data = {
        "init_images": [current_image],
        "inpainting_fill": 0,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 1,
        "inpainting_mask_invert": 1,
        "resize_mode": 0,
        "denoising_strength": denoising_strength,
        "prompt": prompt,
        "negative_prompt": bad_prompt,
        "alwayson_scripts":{
            "ControlNet": {
                "args":[
                        {
                        "input_image": current_image,
                        "module": "canny",
                        "model": "control_v11p_sd15_canny",
                        "weight": 2,
                        "guidance": 1.5
                }]
            },
        },
        "seed": 3189343382,
        "subseed": -1,
        "subseed_strength": -1,
        "sampler_index": "Euler a",
        "batch_size": 1,
        "n_iter": 1,
        "steps": 20,
        "cfg_scale": 7,
        "width": width,
        "height": height,
        "restore_faces": True,
        "include_init_images": True,
    }

    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.content
    else:
        try:
            error_data = response.json()
            print("Error:")
            print(str(error_data))

        except json.JSONDecodeError:
            print(f"Error: Unable to parse JSON error data.")
        return None


def send_request(last_image_path, current_image_path, prompt, bad_prompt, denoising_strength, width, height):
    url = "http://localhost:7860/sdapi/v1/img2img"

    with open(last_image_path, "rb") as f:
        last_image = base64.b64encode(f.read()).decode("utf-8")

    with open(current_image_path, "rb") as b:
        current_image = base64.b64encode(b.read()).decode("utf-8")

    data = {
        "init_images": [current_image],
        "inpainting_fill": 0,
        "inpaint_full_res": True,
        "inpaint_full_res_padding": 1,
        "inpainting_mask_invert": 1,
        "resize_mode": 0,
        "denoising_strength": denoising_strength,
        "prompt": prompt,
        "negative_prompt":   bad_prompt,
        "alwayson_scripts": {
            "ControlNet": {
                "args": [
                        {
                        "input_image": current_image,
                        "module": "canny",
                        "model": "control_v11p_sd15_canny",
                        "weight": 2,
                        "guidance": 1.5
                },
                    {
                        "input_image": last_image,
                        "model": "diff_control_sd15_temporalnet_fp16",
                        "module": "none",
                        "weight": 1.5,
                        "guidance": 1,
                    }

                ]
            }
        },
        "seed": 3189343382,
        "subseed": -1,
        "subseed_strength": -1,
        "sampler_index": "Euler a",
        "batch_size": 1,
        "n_iter": 1,
        "steps":  20,
        "cfg_scale": 7,
        "width": width,
        "height": height,
        "restore_faces": True,
        "include_init_images": True,
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.content
    else:
        try:
            error_data = response.json()
            print("Error:")
            print(str(error_data))

        except json.JSONDecodeError:
            print(f"Error: Unable to parse JSON error data.")
        return None


def set_and_load_checkpoint(name):
    
    data = requests.get(url="http://localhost:7860/sdapi/v1/sd-models").json()

    model_title = "realistic.safetensors [18ed2b6c48]"
    for obj in data:
        if obj["model_name"] == name:
            model_title = obj["title"]


    option_payload = {
        "sd_model_checkpoint": model_title,
        "CLIP_stop_at_last_layers": 2
    }

    response = requests.post(url="http://localhost:7860/sdapi/v1/options", json=option_payload)

    if response.status_code != 200:
        print("Error occured when changing model")



vid_num = int(input("Enter video num to test [0-4]: "))
# random_vid = randint(0,4)

video_name = str(vid_num)+".mp4"

# denoising_strength = choice(denoisingStrengths)
denoising_strength = float(input("Enter denoising strength: "))

# checkpoint = choice(checkpoints)
checkpoint = input("Enter checkpoint: ")

prompt = choice(promptDictionary[checkpoint])

bad_prompt = choice(badPromptDictionary[checkpoint])


test_folder = os.path.join(script_directory,str(uuid.uuid1()))

os.makedirs(test_folder, exist_ok=True)

log_path = os.path.join(test_folder, "log.txt")
with open(log_path, "w") as log_file:
    log_file.write(f"Prompt: {prompt}\n")
    log_file.write(f"Base Model: {checkpoint}\n")
    log_file.write(f"Denoising Strength: {denoising_strength}\n")


video_path = os.path.join(script_directory, "videos", video_name)
y_folder = os.path.join(test_folder, 'Input_Images')

print("Getting Frames...")
extract_frames(video_path, y_folder)
print("Getting OG video durations")
og_duration = get_video_length(video_path)
print(f'Video is {og_duration} seconds long')

y_paths_with_dimensions = get_image_paths(y_folder)
print("Video Aspect Ratio")
print(f'Width: {y_paths_with_dimensions[0][1][0]}')
print(f'Height: {y_paths_with_dimensions[0][1][1]}')

print("Loading Checkpoint...")
set_and_load_checkpoint(checkpoint)
print("Loaded Checkpoint")

start_time = time.time()

print("Getting init image...")
init_image_response = make_init_image(prompt, bad_prompt, denoising_strength,y_paths_with_dimensions[0][1][0], y_paths_with_dimensions[0][1][1], os.path.join(test_folder,"Input_Images","frame-0001.jpg"))
print("Init image recieved and saved")

init_image_path = os.path.join(test_folder, "init.png")
data = json.loads(init_image_response)
encoded_image = data["images"][0]
with open(init_image_path, "wb") as f:
    f.write(base64.b64decode(encoded_image))


x_path = os.path.join(test_folder, "init.png") 

output_folder = os.path.join(test_folder, "output")
os.makedirs(output_folder, exist_ok=True)

output_images = []
output_images.append(send_request(x_path, y_paths_with_dimensions[0][0],prompt, bad_prompt, denoising_strength,
                                    y_paths_with_dimensions[0][1][0], y_paths_with_dimensions[0][1][1]))
output_paths = []

for i in range(1, len(y_paths_with_dimensions)):
    result_image = output_images[i - 1]
    temp_image_path = os.path.join(output_folder, f"temp_image_{i}.png")
    data = json.loads(result_image)
    encoded_image = data["images"][0]
    with open(temp_image_path, "wb") as f:
        f.write(base64.b64decode(encoded_image))
    output_paths.append(temp_image_path)
    result = send_request(temp_image_path, y_paths_with_dimensions[i][0],prompt, bad_prompt, denoising_strength, y_paths_with_dimensions[i][1][0], y_paths_with_dimensions[i][1][1])
    output_images.append(result)
    print(f"Written data for frame {i}:")

elapsed_time = time.time() - start_time
with open(log_path, "a") as log_file:
    log_file.write(f"Time To Generate: {elapsed_time} seconds\n")

clip = ImageSequenceClip(output_paths, fps=fps)
output_video_path = os.path.join(output_folder, "output_video.mp4")
clip.write_videofile(output_video_path)

print("Video created:", output_video_path)