import argparse
import os
from glob import glob

import cv2
import mediapipe as mp
import numpy as np
import torch
from tqdm import tqdm

import constants
from models.data_loader import video_test
from models.difussionsampler import DiffusionSampler
from models.hmr_model import hmr
from models.MANO import MANO
from models.renderer import Renderer
from models.utils import (
    batch_euler2matzxy,
    coordtrans,
    crop2expandsquare_zeros,
    rotmat2eulerzxy,
)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def bbox_extraction(vid_path):
    """
    Extract bounding boxes for hands in a given video.

    Args:
        vid_path: Path to the video file.

    Returns:
    """
    # read video frames
    frame_list = sorted(glob(os.path.join(vid_path, "*")))
    print(f"Number of frames found: {len(frame_list)}")

    # Get video properties
    fps = 30

    frames = []
    frame_timestamps = []
    frame_index = 0

    for frame_path in frame_list:
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        frames.append(frame)

        # Calculate timestamp for this frame in milliseconds
        frame_timestamp_ms = int(frame_index * 1000 / fps)
        frame_timestamps.append(frame_timestamp_ms)
        frame_index += 1

        # Calculate timestamp for this frame in milliseconds
        frame_timestamp_ms = int(frame_index * 1000 / fps)
        frame_timestamps.append(frame_timestamp_ms)
        frame_index += 1

    # Create a hand landmarker instance with the video mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="./assets/hand_landmarker_mediapipe.task"
        ),
        running_mode=VisionRunningMode.VIDEO,
    )

    # extract hand bounding boxes
    hand_bboxes = []
    with HandLandmarker.create_from_options(options) as landmarker:
        for i, frame in enumerate(frames):
            # Convert BGR (OpenCV) to RGB (MediaPipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            hand_landmarker_result = landmarker.detect_for_video(
                mp_image, frame_timestamps[i]
            )

            # Process the results to extract bounding boxes
            if hand_landmarker_result.hand_landmarks:
                for hand_landmarks in hand_landmarker_result.hand_landmarks:
                    # Calculate bounding box from landmarks
                    x_coords = [landmark.x for landmark in hand_landmarks]
                    y_coords = [landmark.y for landmark in hand_landmarks]

                    # Convert normalized coordinates to pixel coordinates
                    h, w = frame.shape[:2]
                    x_min = int(min(x_coords) * w)
                    x_max = int(max(x_coords) * w)
                    y_min = int(min(y_coords) * h)
                    y_max = int(max(y_coords) * h)

                    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
                    box_size = int(1.2 * max(x_max - x_min, y_max - y_min)) // 2

                    # ensure the box is within image bounds
                    x1 = max(0, cx - box_size)
                    y1 = max(0, cy - box_size)
                    x2 = min(w, cx + box_size)
                    y2 = min(h, cy + box_size)

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    w = (x2 - x1) // 2
                    h = (y2 - y1) // 2

                    frame_bboxes = [cx, cy, w, h]

                hand_bboxes.append(frame_bboxes)
            else:
                hand_bboxes.append([])  # No hands detected in this frame

    return frames, hand_bboxes


def pred_extraction(frames, hand_bboxes, output_path, video_name, mano, renderer, local2global, device):
    import torchvision.transforms as transforms

    # load KNOWN_Hand Prediction model
    model = hmr()
    checkpoint = torch.load("assets/KNOWN_HAND.pt")
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    # define image data processing
    trans = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(constants.IMG_NORM_MEAN, constants.IMG_NORM_STD),
        ]
    )

    # define video writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    render_output = cv2.VideoWriter(
        f"{output_path}/{video_name}_framewise.avi",
        fourcc,
        20,
        (int(constants.IMG_RES * 2), constants.IMG_RES),
    )

    # process the video frames
    num_frames = len(frames)
    pred_verts = []
    pred_joints = []
    pred_cams = []
    render_backgrounds = []
    for i in tqdm(range(num_frames)):

        image = frames[i]
        H, W = image.shape[:2]
        bbox = hand_bboxes[i]
        if len(bbox) == 0:
            # TODO: no hand detection needs to be handled
            pred_verts.append(pred_verts[-1])
            output_frame = np.ones(
                (constants.IMG_RES, constants.IMG_RES * 2, 3)
            ).astype(np.uint8)
        else:
            # cut and resize
            X_cropped, offset_Pts, _ = crop2expandsquare_zeros(image, bbox, 0)
            X_croppedresize = cv2.resize(
                X_cropped,
                (constants.IMG_RES, constants.IMG_RES),
                interpolation=cv2.INTER_CUBIC,
            )
            X = trans(X_croppedresize.copy()).float().unsqueeze(0).to(device)

            with torch.no_grad():
                # get prediction for the input image
                pred_all = model(X)
                pred_pose_mean, _, _, pred_beta_mean, _, pred_cam_mean, _ = pred_all

                # obtain 3D hand vertex positions
                pred_pose_rotmat_mean = batch_euler2matzxy(
                    pred_pose_mean.reshape(-1, 3)
                ).view(-1, 16, 3, 3)
                pred_pose_rotmat_mean = coordtrans(
                    pred_pose_rotmat_mean, local2global, 1
                )
                pred_output_mean = mano.forward(
                    betas=pred_beta_mean, thetas_rotmat=pred_pose_rotmat_mean
                )
                pred_vert = pred_output_mean.vertices
                pred_verts.append(pred_vert.detach().cpu().numpy().squeeze())
                pred_joint = pred_output_mean.joints
                pred_joints.append(pred_joint.detach().cpu().numpy().squeeze())

                # render the 3D hand
                pred_cam_t = torch.stack(
                    [
                        pred_cam_mean[:, 1],
                        pred_cam_mean[:, 2],
                        constants.FOCAL_LENGTH / (pred_cam_mean[:, 0] + 1e-9),
                    ],
                    dim=-1,
                )
                pred_cams.append(pred_cam_t.detach().cpu().numpy().squeeze())

                rend_img = renderer.demo(
                    pred_vert[0].detach().cpu().numpy(),
                    np.array([0, 0, 1]),
                    np.ones_like(X_croppedresize[:, :, ::-1] / 255.0),
                )
                output_frame = np.hstack(
                    [X_croppedresize[:, :, ::-1] / 255.0, rend_img]
                )
                render_backgrounds.append(X_croppedresize[:, :, ::-1] / 255.0)

        render_output.write((output_frame[:, :, ::-1] * 255).astype(np.uint8))

    return pred_verts, pred_cams, render_backgrounds

def pred_refinement(hand_preds_frame_wise, pred_cams, render_backgrounds, mano, renderer, local2global, output_path, video_name, device):

    from torch.utils.data import DataLoader

    from models.DIP import DIP

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load DIP model
    model = DIP(device = device, 
                        seqlen = constants.seqlen, 
                        decoder = True,
                        d_model = 512, 
                        nhead = 8, 
                        d_hid = 512,
                        nlayers = 4, 
                        dropout = 0.1).to(device)

    model.load_state_dict(torch.load("assets/DIP-Hand.pth"), strict=True)
    model.eval()

    # define data loader
    data_loader = DataLoader(dataset=video_test(hand_preds_frame_wise, constants.seqlen),
                                        batch_size=64,
                                        shuffle=False)
    
    # define diffusion sampler
    sampler = DiffusionSampler(constants.diff_steps, constants.kappa, device)

    hand_preds_refined = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
            input_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

            # NxT
            src_mask = input_batch['src_mask'].type(torch.bool)

            N, T = src_mask.shape

            # perform inverse kinematics to define y0
            pred_verts = input_batch['pred_verts'].type(torch.float32) # local
            n_batch = (torch.zeros((pred_verts.shape[0],))*0+constants.diff_steps).long()
            _, decoder_pred = model.forward(pred_verts.reshape([-1, constants.seqlen, 778, 3]),
                                                        pred_verts.reshape([-1, constants.seqlen, 778, 3]),
                                                        n_batch.float().to(device), 
                                                        src_mask.clone(), 
                                                        None,
                                                        None,
                                                        mano)
            x0_shape_decoder_pred, _, _, _, x0_rotmat_decoder_pred_t, x0_mano_decoder_pred_t = decoder_pred

            x0_shape_pred = x0_shape_decoder_pred.expand(-1, constants.seqlen, -1)
            x0_rotmat_pred_t = coordtrans(x0_rotmat_decoder_pred_t, local2global.expand(N*T,-1,-1,-1), 0)
            x0_pose_pred = rotmat2eulerzxy(x0_rotmat_pred_t.reshape([-1, 3, 3])).reshape([-1, constants.seqlen, 48])

            # NxTx48
            y0_pose_pred = x0_pose_pred
            # NxTx10 
            y0_shape_pred = x0_shape_pred

            y0_shape_t = y0_shape_pred.reshape(-1,10)
            y0_rotmat = batch_euler2matzxy(y0_pose_pred.reshape(-1,3)).reshape(-1,16,3,3)
            y0_rotmat_t = coordtrans(y0_rotmat, local2global.expand(N*T,-1,-1,-1), 1)
            y0_mano_t = mano.forward(betas=y0_shape_t, thetas_rotmat=y0_rotmat_t)
            y0_vertices_t = y0_mano_t.vertices

            # reverse diffusion  
            for n in range(constants.diff_steps, 1, -1):
                if n == constants.diff_steps:
                    n_batch = torch.zeros((y0_shape_pred.shape[0],))*0+constants.diff_steps

                    xn_shape = y0_shape_pred + constants.kappa * torch.randn_like(y0_shape_pred).to(device)
                    xn_pose = y0_pose_pred + constants.kappa * torch.randn_like(y0_pose_pred).to(device)
                else:
                    # generate diffused data
                    e_shape = y0_shape_pred.clone()-x0_shape_pred
                    e_pose = y0_pose_pred.clone()-x0_pose_pred

                    n_batch = (torch.zeros((y0_shape_pred.shape[0],))*0+n).long()

                    xn_shape = sampler.q_sample(x0_shape_pred, e_shape, n_batch)
                    xn_pose = sampler.q_sample(x0_pose_pred, e_pose, n_batch)

                xn_shape_t = xn_shape.view(-1,10)
                xn_rotmat = batch_euler2matzxy(xn_pose.reshape(-1,3)).reshape(-1,16,3,3)
                xn_rotmat_t = coordtrans(xn_rotmat, local2global.expand(N*T,-1,-1,-1), 1)
                xn_mano_t = mano.forward(betas=xn_shape_t, thetas_rotmat=xn_rotmat_t)
                xn_vertices_t = xn_mano_t.vertices

                _, decoder_pred = model.forward(y0_vertices_t.reshape([-1, constants.seqlen, 778, 3]),
                                                                xn_vertices_t.reshape([-1, constants.seqlen, 778, 3]),
                                                                n_batch.float().to(device), 
                                                                src_mask.clone(), 
                                                                None,
                                                                None,
                                                                mano)

                x0_shape_decoder_pred, _, _, _, x0_rotmat_decoder_pred_t, x0_mano_decoder_pred_t = decoder_pred

                x0_shape_pred = x0_shape_decoder_pred.expand(-1, constants.seqlen, -1)
                x0_rotmat_pred_t = coordtrans(x0_rotmat_decoder_pred_t, local2global.expand(N*T,-1,-1,-1), 0)
                x0_pose_pred = rotmat2eulerzxy(x0_rotmat_pred_t.reshape([-1, 3, 3])).reshape([-1, constants.seqlen, 48])

            pred_vert_refine = x0_mano_decoder_pred_t.vertices[::constants.seqlen].detach().cpu().numpy()
            hand_preds_refined.extend(pred_vert_refine.tolist())

    # define video writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    render_output = cv2.VideoWriter(
        f"{output_path}/{video_name}_refined.avi",
        fourcc,
        20,
        (int(constants.IMG_RES * 2), constants.IMG_RES),
    )
    num_frames = len(hand_preds_refined)
    for i in tqdm(range(num_frames), desc="Rendering frames"):
        rend_img = renderer.demo(
            hand_preds_refined[i],
            np.array([0, 0, 1]),
            np.ones_like(render_backgrounds[i]),
        )
        output_frame = np.hstack(
            [render_backgrounds[i], rend_img]
        )

        render_output.write((output_frame[:, :, ::-1] * 255).astype(np.uint8))

    render_output.release()

    return hand_preds_refined

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to a video frame folder"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="Path to the output folder for processed video",
    )

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    video_name = args.video_path.split("/")[-1].split(".")[0]


    # load MANO model
    mano = MANO("RIGHT", device)

    # define joint rotation transformation
    local2global = (
        torch.from_numpy(constants.euler_coordtrans_RIGHT)
        .to(device)
        .type(torch.float32)
    )
    local2global = (
        batch_euler2matzxy(local2global.view(-1, 3))
        .view(-1, 15, 3, 3)
        .expand(1, -1, -1, -1)
    )

    # define renderer
    renderer = Renderer(
        focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=mano.faces
    )

    # extract frames and hand bounding boxes
    frames, hand_bboxes = bbox_extraction(args.video_path)

    # extract frame-wise hand mesh predictions
    hand_preds_frame_wise, pred_cams, render_backgrounds  = pred_extraction(frames, hand_bboxes, args.output_path, video_name, mano, renderer, local2global, device)

    # generate refined predictions using DIP Hand
    # TODO: taking the missing detection frames into consideration
    hand_preds_refined = pred_refinement(hand_preds_frame_wise, pred_cams, render_backgrounds, mano, renderer, local2global, args.output_path, video_name, device)

    assert len(hand_preds_refined) == len(hand_preds_frame_wise)

if __name__ == "__main__":
    main()
