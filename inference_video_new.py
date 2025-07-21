import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings

warnings.filterwarnings("ignore")


def calculate_target_frame_positions(source_fps, target_fps, total_source_frames):
    """
    Calculate which frames need to be generated for the target frame rate.
    Returns a list of (source_frame_index1, source_frame_index2, interpolation_factor) tuples.
    """
    frame_positions = []

    # Calculate the time duration of the video
    duration = (total_source_frames - 1) / source_fps

    # Calculate number of target frames
    total_target_frames = int(duration * target_fps) + 1

    for target_idx in range(total_target_frames):
        # Calculate the time position of this target frame
        target_time = target_idx / target_fps

        # Calculate the corresponding position in source frames
        source_position = target_time * source_fps

        # Find the two source frames to interpolate between
        source_idx1 = int(source_position)
        source_idx2 = min(source_idx1 + 1, total_source_frames - 1)

        # Calculate interpolation factor (0 means use frame1, 1 means use frame2)
        if source_idx1 == source_idx2:
            interpolation_factor = 0.0
        else:
            interpolation_factor = source_position - source_idx1

        frame_positions.append((source_idx1, source_idx2, interpolation_factor))

    return frame_positions


def main():
    parser = argparse.ArgumentParser(
        description="Non-integer frame rate video interpolation"
    )
    parser.add_argument("--input", required=True, type=str, help="Input video path")
    parser.add_argument("--output", type=str, help="Output video path")
    parser.add_argument(
        "--source_fps", type=float, help="Source FPS (auto-detected if not specified)"
    )
    parser.add_argument("--target_fps", required=True, type=float, help="Target FPS")
    parser.add_argument(
        "--model",
        dest="modelDir",
        type=str,
        default="train_log",
        help="directory with trained model files",
    )
    parser.add_argument(
        "--scale",
        dest="scale",
        type=float,
        default=1.0,
        help="Scale factor for processing",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # Load model
    from train_log.RIFE_HDv3 import Model

    model = Model()
    model.load_model(args.modelDir, -1)
    print("Loaded RIFE HD model.")
    model.eval()
    model.device()

    # Open video and get properties
    cap = cv2.VideoCapture(args.input)
    source_fps = args.source_fps if args.source_fps else cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Input video: {args.input}")
    print(f"Source FPS: {source_fps}, Target FPS: {args.target_fps}")
    print(f"Total source frames: {total_frames}")
    print(f"Resolution: {width}x{height}")

    # Calculate padding for model
    tmp = max(128, int(128 / args.scale))
    ph = ((height - 1) // tmp + 1) * tmp
    pw = ((width - 1) // tmp + 1) * tmp
    padding = (0, pw - width, 0, ph - height)

    # Calculate target frame positions
    frame_positions = calculate_target_frame_positions(
        source_fps, args.target_fps, total_frames
    )
    print(f"Total target frames: {len(frame_positions)}")

    # Setup output video writer
    fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
    if args.output:
        output_path = args.output
    else:
        video_name_wo_ext = os.path.splitext(args.input)[0]
        output_path = f"{video_name_wo_ext}_{source_fps}to{args.target_fps}fps.mp4"

    out = cv2.VideoWriter(output_path, fourcc, args.target_fps, (width, height))

    # Load all frames into memory
    print("Loading source frames...")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Process frames
    print("Generating interpolated frames...")
    pbar = tqdm(total=len(frame_positions))

    for source_idx1, source_idx2, interp_factor in frame_positions:
        if interp_factor == 0.0 or source_idx1 == source_idx2:
            # No interpolation needed, use the source frame directly
            out.write(frames[source_idx1])
        else:
            # Get the two frames to interpolate between
            frame1 = frames[source_idx1]
            frame2 = frames[source_idx2]

            # Convert to tensors (BGR to RGB and normalize)
            I0 = (
                torch.from_numpy(np.transpose(frame1[:, :, ::-1].copy(), (2, 0, 1)))
                .to(device)
                .unsqueeze(0)
                .float()
                / 255.0
            )
            I1 = (
                torch.from_numpy(np.transpose(frame2[:, :, ::-1].copy(), (2, 0, 1)))
                .to(device)
                .unsqueeze(0)
                .float()
                / 255.0
            )

            # Pad images
            I0 = F.pad(I0, padding)
            I1 = F.pad(I1, padding)

            # Perform interpolation
            with torch.no_grad():
                interpolated = model.inference(
                    I0, I1, timestep=interp_factor, scale=args.scale
                )

            # Convert back to numpy and write
            interpolated_frame = (
                (interpolated[0] * 255)
                .byte()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)[:height, :width]
            )
            out.write(interpolated_frame[:, :, ::-1])

        pbar.update(1)

    pbar.close()
    out.release()

    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
