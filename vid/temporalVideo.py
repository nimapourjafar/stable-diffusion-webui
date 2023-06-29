import os
import glob
import requests
import json
from pprint import pprint
import base64
from io import BytesIO
from PIL import Image
from moviepy.editor import ImageSequenceClip, vfx
import subprocess

script_directory = os.path.dirname(os.path.abspath(__file__))
fps = 10
checkpoints = ["disneyPixarCartoon", "realistic", "anime"]
account_name = "@lsoojung"

def get_and_download_random_tiktok(account_name):

    return 

def extract_frames(input_folder, output_folder, filename, fps=10):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Generate the output filename pattern
    output_pattern = os.path.join(output_folder, 'frame-%04d.jpg')

    # Run FFmpeg command to extract frames
    ffmpeg_cmd = [
        'ffmpeg',
        '-i',
        os.path.join(input_folder, filename),
        '-vf',
        'fps={}'.format(fps),
        output_pattern
    ]
    subprocess.run(ffmpeg_cmd)

def get_video_length(input_folder, filename):
    # Run FFprobe command to get video information
    ffprobe_cmd = [
        'ffprobe',
        '-v',
        'error',
        '-show_entries',
        'format=duration',
        '-of',
        'default=noprint_wrappers=1:nokey=1',
        os.path.join(input_folder, filename)
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


def send_request(last_image_path, temp_path, current_image_path, width, height):
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
        "denoising_strength": 0.75,
        "prompt": "20 yo, body, (hi-top fade:1.3), dark theme, soothing tones, muted colors, high contrast, extremely detailed,  (natural skin texture, hyperrealism, soft light, sharp)",
        "negative_prompt": "easynegative,ng_deepnegative_v1_75t,(worst quality:2),(low quality:2),(normal quality:2),lowres, fuzzy background, ugly body, ugly face, bad anatomy,bad hands,normal quality,((monochrome)),((grayscale)),((watermark)),",
        "alwayson_scripts": {
            "ControlNet": {
                "args": [
                    {
                        "input_image": current_image,
                        "module": "hed",
                        "model": "control_hed-fp16",
                        "weight": 1.5,
                        "guidance": 1,
                    },
                    {
                        "input_image": last_image,
                        "model": "diff_control_sd15_temporalnet_fp16",
                        "module": "none",
                        "weight": 0.7,
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
        "steps": 30,
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



set_and_load_checkpoint("realistic")


for i in range(50):

    output_dir_folder_name =  get_and_download_random_tiktok(account_name)

    input_folder = os.path.join(script_directory, output_dir_folder_name)
    output_folder = os.path.join(input_folder, 'Input_Images')
    filename = 'testvid.mp4'

    extract_frames(input_folder, output_folder, filename, fps)
    og_duration = get_video_length(input_folder, filename)

    # make init image

    x_path = os.path.join(output_dir_folder_name, "init.png")  # Update the path to the initial image

    y_folder = os.path.join(output_dir_folder_name, "Input_Images/")  # Update the path to the input images folder

    y_paths_with_dimensions = get_image_paths(y_folder)

    output_folder = os.path.join(output_dir_folder_name, "output")
    os.makedirs(output_folder, exist_ok=True)

    output_images = []
    output_images.append(send_request(x_path, y_folder, y_paths_with_dimensions[0][0],
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
        result = send_request(temp_image_path, y_folder, y_paths_with_dimensions[i][0],
                              y_paths_with_dimensions[i][1][0], y_paths_with_dimensions[i][1][1])
        output_images.append(result)
        print(f"Written data for frame {i}:")

    clip = ImageSequenceClip(output_paths, fps=fps)
    output_video_path = os.path.join(output_folder, "output_video.mp4")
    clip.write_videofile(output_video_path)
    
    print("Video created:", output_video_path)