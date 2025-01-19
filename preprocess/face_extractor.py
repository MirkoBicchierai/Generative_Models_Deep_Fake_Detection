import os
import cv2
import torch
from tqdm import tqdm
import numpy as np
import argparse
from facenet_pytorch import MTCNN

# Define the number of frames to extract from each video
num_frames_to_extract = 10

# Use MTCNN for GPU-accelerated face detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_detector = MTCNN(keep_all=True, device=device)

parser = argparse.ArgumentParser(description="Extract faces from videos.")
parser.add_argument(
    "--root_dir",
    type=str,
    required=True,
    help="Path to the root directory containing manipulated_sequences and original_sequences",
)


def get_boundingbox(box, width, height, scale=1.3, minsize=None):
    """Get enlarged and square bounding box."""
    x1, y1, x2, y2 = box
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        size_bb = max(size_bb, minsize)
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def extract_faces_from_frame(frame, output_dir, video_name, frame_number, face_count):
    # Convert frame to RGB (as required by facenet-pytorch)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]

    # Detect faces using MTCNN
    boxes, _ = face_detector.detect(rgb_frame)

    if boxes is not None and len(boxes) > 0:
        # Find the bounding box with the largest area
        largest_face = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        x1, y1, size = get_boundingbox(largest_face, width, height)
        cropped_face = frame[y1 : y1 + size, x1 : x1 + size]

        # Save cropped face
        face_filename = (
            f"{video_name}_frame_{frame_number:04d}_face_{face_count:04d}.png"
        )
        face_path = os.path.join(output_dir, face_filename)
        cv2.imwrite(face_path, cropped_face)
        face_count += 1

    return face_count


def extract_frames_from_video(video_file, output_dir):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0  # Track frames extracted
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    face_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    random_list = []  # Track already chosen random frame numbers

    while frame_count < num_frames_to_extract:
        # Generate a random frame index
        random_fps = np.random.randint(0, fps)
        while random_fps in random_list:
            random_fps = np.random.randint(0, fps)
        random_list.append(random_fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, random_fps)
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame to extract faces
        face_count = extract_faces_from_frame(
            frame, output_dir, video_name, frame_count, face_count
        )
        frame_count += 1

    cap.release()


def main():
    args = parser.parse_args()
    root_dir = args.root_dir

    forgery_map = {
        "manipulated_sequences/Deepfakes": "DeepFakes",
        "manipulated_sequences/Face2Face": "Face2Face",
        "manipulated_sequences/NeuralTextures": "NeuralTextures",
        "manipulated_sequences/FaceShifter": "FaceShifter",
        "manipulated_sequences/FaceSwap": "FaceSwap",
        "original_sequences/youtube": "youtube",
    }

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for key, forgery_name in forgery_map.items():
            video_counter = 0
            if key in dirpath:
                print(f"Processing {dirpath}")

                suffix = (
                    "manipulated" if "manipulated_sequences" in dirpath else "original"
                )
                output_base_dir = os.path.join(root_dir, "output", forgery_name)
                os.makedirs(output_base_dir, exist_ok=True)

                # Determine the split directories based on video_counter

                with tqdm(total=len(filenames)) as pbar:
                    for filename in sorted(filenames):
                        if video_counter < 720:
                            split = "train"
                        elif video_counter < 860:
                            split = "val"
                        else:
                            split = "test"

                        output_dir = os.path.join(output_base_dir, split, suffix)
                        os.makedirs(output_dir, exist_ok=True)

                        if filename.endswith((".mp4", ".avi")):
                            video_file = os.path.join(dirpath, filename)
                            extract_frames_from_video(video_file, output_dir)
                            pbar.update(1)
                            video_counter += 1

    print("Frame and face extraction complete.")


if __name__ == "__main__":
    main()
