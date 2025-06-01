import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr
from refacer import Refacer
import argparse
import ngrok
import imageio
import numpy as np
from PIL import Image
import tempfile
import base64
import pyfiglet
import shutil
import time
import cv2
import ffmpeg

print("\033[94m" + pyfiglet.Figlet(font='slant').renderText("NeoRefacer") + "\033[0m")

def cleanup_temp(folder_path):
    try:
        shutil.rmtree(folder_path)
        print("Gradio cache cleared successfully.")
    except Exception as e:
        print(f"Error: {e}")

# Prepare temp folder
os.environ["GRADIO_TEMP_DIR"] = "./tmp"
if os.path.exists("./tmp"):
    cleanup_temp(os.environ['GRADIO_TEMP_DIR'])
if not os.path.exists("./tmp"):
    os.makedirs("./tmp")

# Parse arguments
parser = argparse.ArgumentParser(description='Refacer')
parser.add_argument("--max_num_faces", type=int, default=8)
parser.add_argument("--force_cpu", default=False, action="store_true")
parser.add_argument("--share_gradio", default=False, action="store_true")
parser.add_argument("--server_name", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=7860)
parser.add_argument("--colab_performance", default=False, action="store_true")
parser.add_argument("--ngrok", type=str, default=None)
parser.add_argument("--ngrok_region", type=str, default="us")
args = parser.parse_args()

# Initialize
refacer = Refacer(force_cpu=args.force_cpu, colab_performance=args.colab_performance)
num_faces = args.max_num_faces

def create_dummy_image():
    dummy = Image.new('RGB', (1, 1), color=(255, 255, 255))
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir="./tmp", suffix=".png")
    dummy.save(temp_file.name)
    return temp_file.name

def run_image(*vars):
    image_path = vars[0]
    origins = vars[1:(num_faces+1)]
    destinations = vars[(num_faces+1):(num_faces*2)+1]
    thresholds = vars[(num_faces*2)+1:-3] # New slice
    face_mode = vars[-3] # New index
    partial_reface_ratio = vars[-2] # New index
    enhance_quality = vars[-1] # New variable

    disable_similarity = (face_mode in ["Single Face", "Multiple Faces"])
    multiple_faces_mode = (face_mode == "Multiple Faces")

    faces = []
    for k in range(num_faces):
        if destinations[k] is not None:
            faces.append({
                'origin': origins[k] if not multiple_faces_mode else None,
                'destination': destinations[k],
                'threshold': thresholds[k] if not multiple_faces_mode else 0.0
            })

    return refacer.reface_image(
        image_path, 
        faces, 
        disable_similarity=disable_similarity, 
        multiple_faces_mode=multiple_faces_mode, 
        partial_reface_ratio=partial_reface_ratio,
        enhance_quality=enhance_quality # Pass the new flag
    )

def run(*vars): # Video processing
    video_path = vars[0]
    origins = vars[1:(num_faces+1)]
    destinations = vars[(num_faces+1):(num_faces*2)+1]
    thresholds = vars[(num_faces*2)+1:-4] # New slice
    preview = vars[-4] # New index
    face_mode = vars[-3] # New index
    partial_reface_ratio = vars[-2] # New index
    enhance_quality = vars[-1] # New variable

    disable_similarity = (face_mode in ["Single Face", "Multiple Faces"])
    multiple_faces_mode = (face_mode == "Multiple Faces")

    faces = []
    for k in range(num_faces):
        if destinations[k] is not None:
            faces.append({
                'origin': origins[k] if not multiple_faces_mode else None,
                'destination': destinations[k],
                'threshold': thresholds[k] if not multiple_faces_mode else 0.0
            })

    mp4_path, gif_path = refacer.reface(
        video_path,
        faces,
        preview=preview,
        disable_similarity=disable_similarity,
        multiple_faces_mode=multiple_faces_mode,
        partial_reface_ratio=partial_reface_ratio,
        enhance_quality=enhance_quality # Pass the new flag
    )
    return mp4_path, gif_path if gif_path else None

def load_first_frame(filepath):
    if filepath is None:
        return None
    frames = imageio.get_reader(filepath)
    return frames.get_data(0)

def extract_faces_auto(filepath, refacer_instance, max_faces=5, isvideo=False):
    if filepath is None:
        return [None] * max_faces

    if isvideo and os.path.getsize(filepath) > 5 * 1024 * 1024:
        print("Video too large for auto-extract, skipping face extraction.")
        return [None] * max_faces

    frame = load_first_frame(filepath)
    if frame is None:
        return [None] * max_faces

    while len(frame.shape) > 3:
        frame = frame[0]

    if frame.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be 3 (RGB), but got {frame.shape[-1]}")

    temp_image_path = os.path.join("./tmp", f"temp_face_extract_{int(time.time() * 1000)}.png")
    Image.fromarray(frame).save(temp_image_path)

    try:
        faces = refacer_instance.extract_faces_from_image(temp_image_path, max_faces=max_faces)
        return faces + [None] * (max_faces - len(faces))
    finally:
        if os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_image_path}: {e}")

def toggle_tabs_and_faces(mode, face_tabs, origin_faces):
    if mode == "Single Face":
        tab_updates = [gr.update(visible=(i == 0)) for i in range(len(face_tabs))]
        origin_updates = [gr.update(visible=False) for _ in range(len(origin_faces))]
    elif mode == "Multiple Faces":
        tab_updates = [gr.update(visible=True) for _ in range(len(face_tabs))]
        origin_updates = [gr.update(visible=False) for _ in range(len(origin_faces))]
    else:
        tab_updates = [gr.update(visible=True) for _ in range(len(face_tabs))]
        origin_updates = [gr.update(visible=True) for _ in range(len(origin_faces))]
    return tab_updates + origin_updates
    
def handle_tif_preview(filepath):
    if filepath is None:
        return None
    preview_path = os.path.join("./tmp", f"tif_preview_{int(time.time() * 1000)}.jpg")
    Image.open(filepath).convert('RGB').save(preview_path)
    return preview_path

# refacer and num_faces are defined globally in app.py

def get_image_info(image_path):
    if image_path is None: # Handle case where image is cleared or not provided
        return "N/A", "N/A", "N/A"
    try:
        # For gr.Image(type="filepath"), image_path should be a string path.
        img = Image.open(image_path)
        resolution = f"{img.width} x {img.height}"
        img.close() 
        
        size_bytes = os.path.getsize(image_path)
        size_mb = f"{size_bytes / (1024 * 1024):.2f} MB"
        
        filename = os.path.basename(image_path)
        
        return filename, resolution, size_mb
    except FileNotFoundError:
        return "File not found", "N/A", "N/A"
    except Exception as e:
        print(f"Error getting image info for {image_path}: {e}")
        return "Error processing file", "Error", "Error"

def update_image_inputs_and_info(filepath):
    if filepath is None: 
        empty_faces = [None] * num_faces # num_faces is global
        return empty_faces + [0.0, "N/A", "N/A", "N/A"]

    # refacer is global
    extracted_faces = extract_faces_auto(filepath, refacer, max_faces=num_faces) 
    name, res, size = get_image_info(filepath)
    
    # Ensure extracted_faces is always a list of length num_faces
    if not isinstance(extracted_faces, list):
        extracted_faces = [None] * num_faces
    elif len(extracted_faces) < num_faces:
        # Pad with None if fewer faces than num_faces slots are returned
        extracted_faces.extend([None] * (num_faces - len(extracted_faces)))
    elif len(extracted_faces) > num_faces:
        # Truncate if more faces than num_faces slots are returned
        extracted_faces = extracted_faces[:num_faces]

    return extracted_faces + [0.0, name, res, size]


def get_video_info(video_path):
    if video_path is None: # Handle case where video is cleared or not provided
        return "N/A", "N/A", "N/A", "N/A"
    try:
        # For gr.Video(type="filepath"), video_path should be a string path.
        filename = os.path.basename(video_path)
        
        size_bytes = os.path.getsize(video_path)
        size_mb = f"{size_bytes / (1024 * 1024):.2f} MB"
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            # Try ffprobe if cv2 fails to open
            try:
                probe = ffmpeg.probe(video_path)
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                if video_stream:
                    width = int(video_stream['width'])
                    height = int(video_stream['height'])
                    resolution = f"{width} x {height}"
                    duration_val = "N/A"
                    if 'duration' in video_stream:
                        duration_val = f"{float(video_stream['duration']):.2f} s"
                    elif 'tags' in video_stream and 'DURATION' in video_stream['tags']:
                        dur_str = video_stream['tags']['DURATION']
                        h, m, s_ms = dur_str.split(':')
                        s, ms_val = map(float, s_ms.split('.'))
                        total_seconds = int(h) * 3600 + int(m) * 60 + s + float(ms_val)/1000
                        duration_val = f"{total_seconds:.2f} s"
                    return filename, resolution, size_mb, duration_val
                else:
                    return filename, "Error opening video", size_mb, "N/A"
            except Exception as e_ff:
                print(f"cv2 & ffprobe failed for {video_path}: {e_ff}")
                return filename, "Error opening video", size_mb, "N/A"

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resolution = f"{width} x {height}"
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = "N/A"
        if fps and fps > 0 and frame_count and frame_count > 0: 
            duration = f"{frame_count / fps:.2f} s"
        else: # Fallback to ffprobe for duration if cv2 specific frame count/fps is problematic
            try:
                probe = ffmpeg.probe(video_path)
                video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                if video_stream:
                    if 'duration' in video_stream:
                        duration = f"{float(video_stream['duration']):.2f} s"
                    elif 'tags' in video_stream and 'DURATION' in video_stream['tags']:
                        dur_str = video_stream['tags']['DURATION']
                        h, m, s_ms = dur_str.split(':')
                        s, ms_val = map(float, s_ms.split('.'))
                        total_seconds = int(h) * 3600 + int(m) * 60 + s + float(ms_val)/1000
                        duration = f"{total_seconds:.2f} s"
            except Exception as e_ff:
                print(f"ffprobe fallback for duration failed for {video_path}: {e_ff}")


        cap.release()
        return filename, resolution, size_mb, duration
        
    except FileNotFoundError:
        return "File not found", "N/A", "N/A", "N/A"
    except Exception as e:
        print(f"Error getting video info for {video_path}: {e}")
        filename_val = os.path.basename(video_path) if video_path and os.path.exists(video_path) else "Error"
        size_mb_val = f"{os.path.getsize(video_path) / (1024 * 1024):.2f} MB" if video_path and os.path.exists(video_path) else "Error"
        return filename_val, "Error processing", size_mb_val, "Error"


def update_video_inputs_and_info(filepath):
    if filepath is None:
        empty_faces = [None] * num_faces 
        return empty_faces + [0.0, "N/A", "N/A", "N/A", "N/A"]

    extracted_faces = extract_faces_auto(filepath, refacer, max_faces=num_faces, isvideo=True) 
    name, res, size, dur = get_video_info(filepath)
    
    if not isinstance(extracted_faces, list): 
        extracted_faces = [None] * num_faces
    elif len(extracted_faces) < num_faces:
        extracted_faces.extend([None] * (num_faces - len(extracted_faces)))
    elif len(extracted_faces) > num_faces:
        extracted_faces = extracted_faces[:num_faces]
        
    return extracted_faces + [0.0, name, res, size, dur]

# --- UI ---
theme = gr.themes.Base(primary_hue="blue", secondary_hue="cyan")

with gr.Blocks(theme=theme, title="NeoRefacer - AI Refacer") as demo:
    with open("icon.png", "rb") as f:
        icon_data = base64.b64encode(f.read()).decode()
    icon_html = f'<img src="data:image/png;base64,{icon_data}" style="width:40px;height:40px;margin-right:10px;">'

    with gr.Row():
        gr.Markdown(f"""
        <div style="display: flex; align-items: center;">
        {icon_html}
        <span style="font-size: 2em; font-weight: bold; color:#2563eb;">NeoRefacer</span>
        </div>
        """)

    # --- IMAGE MODE ---
    with gr.Tab("Image Mode"):
        with gr.Row():
            image_input = gr.Image(label="Original image", type="filepath", file_types=['.jpeg', '.jpg', '.png'])
            image_output = gr.Image(label="Refaced image", interactive=False, type="filepath")
        
        with gr.Row():
            image_filename_display = gr.Textbox(label="File Name", interactive=False)
            image_resolution_display = gr.Textbox(label="Resolution", interactive=False)
            image_size_display = gr.Textbox(label="File Size (MB)", interactive=False)

        with gr.Row():
            face_mode_image = gr.Radio(["Single Face", "Multiple Faces", "Faces By Match"], value="Single Face", label="Replacement Mode")
            partial_reface_ratio_image = gr.Slider(label="Reface Ratio (0 = Full Face, 0.5 = Half Face)", minimum=0.0, maximum=0.5, value=0.0, step=0.1)
            enhance_quality_image = gr.Checkbox(label="Enhance Quality", value=False)
            image_btn = gr.Button("Reface Image", variant="primary")

        origin_image, destination_image, thresholds_image, face_tabs_image = [], [], [], []

        for i in range(num_faces):
            with gr.Tab(f"Face #{i+1}") as tab:
                with gr.Row():
                    origin = gr.Image(label="Face to replace")
                    destination = gr.Image(label="Destination face")
                threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.2)
            origin_image.append(origin)
            destination_image.append(destination)
            thresholds_image.append(threshold)
            face_tabs_image.append(tab)

        face_mode_image.change(fn=lambda mode: toggle_tabs_and_faces(mode, face_tabs_image, origin_image), inputs=[face_mode_image], outputs=face_tabs_image + origin_image)
        demo.load(fn=lambda: toggle_tabs_and_faces("Single Face", face_tabs_image, origin_image), inputs=None, outputs=face_tabs_image + origin_image)

        image_btn.click(fn=run_image, inputs=[image_input] + origin_image + destination_image + thresholds_image + [face_mode_image, partial_reface_ratio_image, enhance_quality_image], outputs=[image_output])
        image_input.change(fn=update_image_inputs_and_info, inputs=image_input, outputs=origin_image + [partial_reface_ratio_image, image_filename_display, image_resolution_display, image_size_display])


    # --- VIDEO MODE ---
    with gr.Tab("Video Mode"):
        with gr.Row():
            video_input = gr.Video(label="Original video", format="mp4", file_types=['.mp4', '.mov', '.avi', '.mkv'])
            video_output = gr.Video(label="Refaced Video", interactive=False, format="mp4")

        with gr.Row():
            video_filename_display = gr.Textbox(label="File Name", interactive=False)
            video_resolution_display = gr.Textbox(label="Resolution", interactive=False)
            video_size_display = gr.Textbox(label="File Size (MB)", interactive=False)
            video_duration_display = gr.Textbox(label="Duration (s)", interactive=False)
            
        with gr.Row():
            face_mode_video = gr.Radio(
                choices=["Single Face", "Multiple Faces", "Faces By Match"],
                value="Single Face",
                label="Replacement Mode"
            )
            partial_reface_ratio_video = gr.Slider(label="Reface Ratio (0 = Full Face, 0.5 = Half Face)", minimum=0.0, maximum=0.5, value=0.0, step=0.1)
            enhance_quality_video = gr.Checkbox(label="Enhance Quality", value=False)
            video_btn = gr.Button("Reface Video", variant="primary")

        preview_checkbox_video = gr.Checkbox(label="Preview Generation (skip 90% of frames)", value=False)

        origin_video, destination_video, thresholds_video, face_tabs_video = [], [], [], []

        for i in range(num_faces):
            with gr.Tab(f"Face #{i+1}") as tab:
                with gr.Row():
                    origin = gr.Image(label="Face to replace")
                    destination = gr.Image(label="Destination face")
                threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.2)
            origin_video.append(origin)
            destination_video.append(destination)
            thresholds_video.append(threshold)
            face_tabs_video.append(tab)

        face_mode_video.change(
            fn=lambda mode: toggle_tabs_and_faces(mode, face_tabs_video, origin_video),
            inputs=[face_mode_video],
            outputs=face_tabs_video + origin_video
        )

        demo.load(
            fn=lambda: toggle_tabs_and_faces("Single Face", face_tabs_video, origin_video),
            inputs=None,
            outputs=face_tabs_video + origin_video
        )
        
        video_input.change(
            fn=update_video_inputs_and_info,
            inputs=video_input,
            outputs=origin_video + [partial_reface_ratio_video, video_filename_display, video_resolution_display, video_size_display, video_duration_display]
        )

        video_btn.click(
            fn=lambda *args: run(*args),
            inputs=[video_input] + origin_video + destination_video + thresholds_video + [preview_checkbox_video, face_mode_video, partial_reface_ratio_video, enhance_quality_video],
            outputs=[video_output, gr.File(visible=False)]
        )

# --- ngrok connect (optional) ---
if args.ngrok:
    def connect(token, port, options):
        try:
            public_url = ngrok.connect(f"127.0.0.1:{port}", **options).url()
            print(f'ngrok URL: {public_url}')
        except Exception as e:
            print(f'ngrok connection aborted: {e}')

    connect(args.ngrok, args.server_port, {'region': args.ngrok_region, 'authtoken_from_env': False})

# --- Launch app ---
demo.queue().launch(favicon_path="icon.png", show_error=True, share=args.share_gradio, server_name=args.server_name, server_port=args.server_port)
