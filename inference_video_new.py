import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import imageio_ffmpeg as ffmpeg
from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")


def transferAudio(sourceVideo, targetVideo):
    import shutil

    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:
        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system(
            '{} -y -i "{}" -c:a copy -vn {}'.format(
                ffmpeg.get_ffmpeg_exe(), sourceVideo, tempAudioFileName
            )
        )

    targetNoAudio = (
        os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    )
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system(
        '{} -y -i "{}" -i {} -c copy "{}"'.format(
            ffmpeg.get_ffmpeg_exe(), targetNoAudio, tempAudioFileName, targetVideo
        )
    )

    if (
        os.path.getsize(targetVideo) == 0
    ):  # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system(
            '{} -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(
                ffmpeg.get_ffmpeg_exe(), sourceVideo, tempAudioFileName
            )
        )
        os.system(
            '{} -y -i "{}" -i {} -c copy "{}"'.format(
                ffmpeg.get_ffmpeg_exe(), targetNoAudio, tempAudioFileName, targetVideo
            )
        )
        if (
            os.path.getsize(targetVideo) == 0
        ):  # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print(
                "Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead."
            )

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")


def pad_image(img, padding, fp16):
    if fp16:
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)


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
    print(frame_positions)
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
        "--fp16",
        dest="fp16",
        action="store_true",
        help="fp16 mode for faster and more lightweight inference on cards with Tensor Cores",
    )
    parser.add_argument(
        "--UHD", dest="UHD", action="store_true", help="support 4k video"
    )
    parser.add_argument(
        "--scale",
        dest="scale",
        type=float,
        default=1.0,
        help="Try scale=0.5 for 4k video",
    )
    parser.add_argument(
        "--skip_static",
        action="store_true",
        help="Skip interpolation for static frames (frames with high similarity)",
    )

    args = parser.parse_args()

    # Adjust scale for UHD
    if args.UHD and args.scale == 1.0:
        args.scale = 0.5
    assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if args.fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    # Load model
    from train_log.RIFE_HDv3 import Model

    model = Model()
    if not hasattr(model, "version"):
        model.version = 0
    model.load_model(args.modelDir, -1)
    print("Loaded 3.x/4.x HD model.")
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
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    if args.output:
        output_path = args.output
    else:
        video_name_wo_ext = os.path.splitext(args.input)[0]
        output_path = f"{video_name_wo_ext}_{source_fps}to{args.target_fps}fps.mp4"

    out = cv2.VideoWriter(output_path, fourcc, args.target_fps, (width, height))

    # Load all frames into memory (for easier random access)
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

            # Convert to tensors (make copy to avoid negative stride issue)
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
            I0 = pad_image(I0, padding, args.fp16)
            I1 = pad_image(I1, padding, args.fp16)

            # Skip interpolation for static frames if requested
            if args.skip_static:
                I0_small = F.interpolate(
                    I0, (32, 32), mode="bilinear", align_corners=False
                )
                I1_small = F.interpolate(
                    I1, (32, 32), mode="bilinear", align_corners=False
                )
                ssim_val = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

                if ssim_val > 0.996:  # Very similar frames
                    # Use simple blending instead of neural interpolation
                    alpha = interp_factor
                    blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
                    out.write(blended)
                    pbar.update(1)
                    continue

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

    # Transfer audio
    print("Transferring audio...")
    try:
        transferAudio(args.input, output_path)
        print(f"Output saved to: {output_path}")
    except Exception as e:
        print(f"Audio transfer failed: {e}")
        print(f"Output saved to: {output_path} (without audio)")


if __name__ == "__main__":
    main()
