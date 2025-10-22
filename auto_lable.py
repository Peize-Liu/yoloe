import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

from ultralytics import YOLOE
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def init_model(model_id: str) -> YOLOE:
    filename = f"{model_id}-seg.pt"
    # Prefer offline cache if available to avoid network dependency
    try:
        path = hf_hub_download(repo_id="jameslahm/yoloe", filename=filename, local_files_only=True)
    except LocalEntryNotFoundError:
        # Fallback to normal download if cache miss and network is available
        path = hf_hub_download(repo_id="jameslahm/yoloe", filename=filename)
    model = YOLOE(path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


class SegPredictorWithCoeff(SegmentationPredictor):
    """Segmentation predictor that also exposes per-instance mask coefficients as `features` on Results."""

    def postprocess(self, preds, img, orig_imgs):  # override to attach coeffs
        p = ops.non_max_suppression(
            preds[0], self.args.conf, self.args.iou, agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det, nc=len(self.model.names), classes=self.args.classes,
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results: List[Results] = []
        proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
        for i, (pred, orig_img, img_path) in enumerate(zip(p, orig_imgs, self.batch[0])):
            if not len(pred):
                masks = None
                feats = np.zeros((0, getattr(self.model.model[-1], "nm", 32)), dtype=np.float32)
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])
                feats = pred[:, 6:].detach().cpu().numpy()
            else:
                masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                feats = pred[:, 6:].detach().cpu().numpy()

            r = Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
            setattr(r, "features", feats)
            results.append(r)
        return results


def collect_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])


def run_auto_labeling(
    source_dir: str,
    output_dir: str,
    names: List[str],
    model_id: str = "yoloe-v8l",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
):
    """
    Runs YOLOE-based auto-labeling on a folder of images and saves segmentation masks and features.

    Args:
        source_dir (str): Path to the folder containing input images.
        output_dir (str): Path to the folder where output .npy files will be saved.
        names (List[str]): A list of text prompts (class names) to detect.
        model_id (str): The model ID to use (e.g., "yoloe-v8l").
        imgsz (int): Image size for inference.
        conf (float): Confidence threshold for detection.
        iou (float): IoU threshold for NMS.
    """
    src_path = Path(source_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model = init_model(model_id)
    model.set_classes(names, model.get_text_pe(names))

    # Feature channel size equals number of mask coefficients
    nm = getattr(model.model.model[-1], "nm", 32)

    image_paths = collect_images(src_path)
    for img_path in tqdm(image_paths, desc=f"Processing {src_path.name}"):
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
            predictor=SegPredictorWithCoeff,
            retina_masks=True,
        )
        res: Results = results[0]

        H, W = res.orig_img.shape[:2]
        seg = np.zeros((H, W), dtype=np.int32)
        sem = np.zeros((nm, H, W), dtype=np.float32)

        if res.boxes is not None and res.masks is not None and len(res.boxes) > 0:
            boxes = res.boxes.data.cpu().numpy()  # [N,6]: xyxy, conf, cls
            masks = res.masks.data  # [N,H,W] torch or np
            masks = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
            feats = getattr(res, "features", None)
            feats = feats if isinstance(feats, np.ndarray) else np.zeros((masks.shape[0], nm), dtype=np.float32)

            # Sort by confidence desc
            order = np.argsort(-boxes[:, 4])
            for idx in order:
                cls_id = int(boxes[idx, 5])  # 0..N-1
                label_id = cls_id + 1  # 1..N
                m = masks[idx] > 0.5
                # Fill seg only where currently 0
                write_mask = m & (seg == 0)
                if not np.any(write_mask):
                    continue
                seg[write_mask] = label_id
                # Broadcast feature vector to CHW at masked positions
                fv = feats[idx].astype(np.float32)
                sem[:, write_mask] = fv[:, None]

        stem = img_path.stem
        np.save(out_path / f"seg_{stem}.npy", seg)
        np.save(out_path / f"sem_{stem}.npy", sem)

if __name__=="__main__":
    # Base directory containing all sequence folders (00, 01, ..., 21)
    base_sequences_dir = "/vepfs-moore-one/peize/Occ-dataset/kitti/dataset/sequences/"
    
    # Common settings for all sequences
    class_names = ["person","car", "bike", "ground", "road", "road sign", "grass", "lawn", "bush", "hedge", "tree", "tree farm", "rock", "creek", "building", "wall", "sky"]
    model_identifier = "yoloe-11l"
    image_size = 1280
    confidence_threshold = 0.2
    iou_threshold = 0.8

    # Loop over all sequence numbers from 00 to 21
    for seq_num in tqdm(range(22), desc="Processing All Sequences"):
        seq_str = f"{seq_num:02d}"
        source_folder = os.path.join(base_sequences_dir, seq_str, "image_2")
        output_folder = os.path.join(base_sequences_dir, seq_str, "yoloe_semantic")

        # Check if the source image folder exists before running
        if not os.path.isdir(source_folder):
            print(f"Warning: Source folder not found, skipping: {source_folder}")
            continue

        print(f"\n--- Starting sequence {seq_str} ---")
        # Call the function to start the process for the current sequence
        run_auto_labeling(
            source_dir=source_folder,
            output_dir=output_folder,
            names=class_names,
            model_id=model_identifier,
            imgsz=image_size,
            conf=confidence_threshold,
            iou=iou_threshold,
        )
    
    print("\n--- All sequences processed. ---")