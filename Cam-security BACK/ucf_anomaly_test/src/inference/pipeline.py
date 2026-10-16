"""
End-to-end inference pipeline: video file -> anomaly report.

Steps:
    1. Extract frames from input video using OpenCV
    2. Build sliding-window clips (same as training)
    3. Run ResNet50 to get features for each frame
    4. Run LSTM to score each clip
    5. Flag clips above threshold as anomalous
    6. Run YOLOv8 on the peak frame of each flagged clip
    7. Return structured results: timestamps + scores + detections
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass, field
from typing import List, Optional
import time

from config import (
    DEVICE, CLIP_LENGTH, CLIP_STRIDE, ANOMALY_THRESHOLD,
    FRAME_SIZE, CHECKPOINTS_DIR, FRAME_INTERVAL
)
from src.models.feature_extractor import get_extractor
from src.models.lstm_model import get_model
from src.models.detector import get_detector, Detection
from src.explainability.gradcam import GradCAM


TRANSFORM = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@dataclass
class ClipResult:
    clip_idx:       int
    start_frame:    int
    end_frame:      int
    timestamp_sec:  float          # start time in seconds
    anomaly_score:  float
    is_anomalous:   bool
    detections:     List[Detection] = field(default_factory=list)
    explanation:    str = ""
    heatmap:        object = None  # numpy [H, W] Grad-CAM heatmap
    overlay_image:  object = None  # PIL Image with heatmap overlay


@dataclass
class VideoResult:
    video_path:     str
    fps:            float
    total_frames:   int
    duration_sec:   float
    clips_analyzed: int
    anomalous_clips: int
    max_score:      float
    results:        List[ClipResult] = field(default_factory=list)
    processing_time: float = 0.0


class SurveillancePipeline:
    """
    Full inference pipeline: video -> anomaly report.

    Load once, run on many videos.
    """

    def __init__(self,
                 checkpoint_path: Optional[Path] = None,
                 yolo_size: str = "n",
                 threshold: float = ANOMALY_THRESHOLD,
                 batch_size: int = 64,
                 verbose: bool = False):

        if checkpoint_path is None:
            checkpoint_path = CHECKPOINTS_DIR / "best_model.pt"

        print("Loading SurveillancePipeline...")

        # ResNet50 feature extractor (frozen)
        self.extractor = get_extractor(DEVICE)

        # LSTM anomaly scorer
        self.lstm = get_model(DEVICE)
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        self.lstm.load_state_dict(ckpt["model_state"])
        self.lstm.eval()
        print(f"  LSTM loaded from epoch {ckpt['epoch']} (val AUC={ckpt['val_auc']:.4f})")

        # YOLOv8 object detector
        self.detector = get_detector(model_size=yolo_size)

        # Grad-CAM (shares the same extractor)
        self.gradcam  = GradCAM(self.extractor)

        self.threshold  = threshold
        self.batch_size = batch_size
        self.verbose    = verbose
        print("Pipeline ready.\n")

    def _extract_frames(self, video_path: str):
        """Read frames from video, sampling every FRAME_INTERVAL frames.

        Matches the training-time sampling rate so features are in-distribution.
        Without sampling, a 40s/30fps video = 1200 frames → minutes on CPU.
        With FRAME_INTERVAL=10, same video = 120 frames → seconds.

        GIFs are handled separately via Pillow:
        - cv2 divides GIF timestamps incorrectly when FRAME_INTERVAL is applied.
        - Pillow reads true per-frame durations from the GIF header, giving the
          correct FPS and total duration.
        - GIF frames are already at a low rate (typically 10 fps), so they are
          sampled with FRAME_INTERVAL the same way as regular videos; effective_fps
          is then gif_fps / FRAME_INTERVAL.
        """
        suffix = Path(video_path).suffix.lower()

        if suffix == ".gif":
            return self._extract_frames_gif(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        idx    = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % FRAME_INTERVAL == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            idx += 1

        cap.release()

        # If the video is too short after subsampling, fall back to extracting
        # every raw frame so we have the best chance of forming a full clip.
        if len(frames) < CLIP_LENGTH and idx > 0:
            cap2 = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap2.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            cap2.release()
            # effective_fps stays at the original fps (no subsampling applied)
            effective_fps = fps
            return frames, effective_fps, idx

        # Report effective fps after sampling so timestamps stay correct
        effective_fps = fps / FRAME_INTERVAL
        return frames, effective_fps, total_frames

    def _extract_frames_gif(self, video_path: str):
        """Extract frames from a GIF using Pillow.

        Why not cv2 for GIFs:
            cv2.VideoCapture on a GIF reports the wrong effective FPS when
            FRAME_INTERVAL is applied because cv2 does not read per-frame
            timing from the GIF header.  Pillow does.

        Sampling: GIF frames are sampled with FRAME_INTERVAL (same as regular
        videos) to keep temporal density consistent with training features.
        effective_fps = gif_true_fps / FRAME_INTERVAL.

        If the GIF has fewer frames than CLIP_LENGTH after sampling, all raw
        frames are returned instead (no subsampling), with effective_fps equal
        to gif_true_fps.
        """
        gif = Image.open(video_path)

        # Read all frames and their per-frame durations from the GIF header
        raw_frames     = []
        frame_duration = []  # milliseconds per frame
        try:
            while True:
                raw_frames.append(gif.copy().convert("RGB"))
                dur = gif.info.get("duration", 100)  # default 100 ms if missing
                frame_duration.append(dur)
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

        n_raw      = len(raw_frames)
        total_ms   = sum(frame_duration)
        avg_ms     = total_ms / n_raw if n_raw else 100.0
        gif_fps    = 1000.0 / avg_ms  # e.g. 100 ms/frame → 10 fps

        # Sample with FRAME_INTERVAL — same policy as regular videos
        sampled = [f for i, f in enumerate(raw_frames) if i % FRAME_INTERVAL == 0]

        # Fallback: if too few frames after sampling, use all raw frames
        if len(sampled) < CLIP_LENGTH:
            sampled       = raw_frames
            effective_fps = gif_fps            # no subsampling → native fps
        else:
            effective_fps = gif_fps / FRAME_INTERVAL

        return sampled, effective_fps, n_raw

    def _frames_to_features(self, frames: List[Image.Image]) -> np.ndarray:
        """Run all frames through ResNet50 in batches. Returns [N, 2048]."""
        all_features = []

        for i in range(0, len(frames), self.batch_size):
            batch_pil = frames[i : i + self.batch_size]
            tensors   = torch.stack([TRANSFORM(f) for f in batch_pil]).to(DEVICE)

            with torch.no_grad():
                feats = self.extractor(tensors)   # [B, 2048]

            all_features.append(feats.cpu().numpy())

        result = np.concatenate(all_features, axis=0)   # [N, 2048]

        if self.verbose:
            print(f"  [diag] features shape : {result.shape}")
            print(f"  [diag] feat  min={result.min():.4f}  max={result.max():.4f}"
                  f"  mean={result.mean():.4f}  std={result.std():.4f}")
            norms = np.linalg.norm(result, axis=1)
            print(f"  [diag] L2 norm per frame — "
                  f"min={norms.min():.2f}  max={norms.max():.2f}  mean={norms.mean():.2f}")

        return result

    def _score_clips(self, features: np.ndarray) -> List[tuple]:
        """
        Build sliding-window clips and score each with LSTM.

        Returns list of (start_idx, end_idx, score).
        """
        N = len(features)
        clips = []

        for start in range(0, N - CLIP_LENGTH + 1, CLIP_STRIDE):
            end  = start + CLIP_LENGTH
            clip = features[start:end]   # [16, 2048]
            clips.append((start, end, clip))

        if not clips:
            return []

        # Batch all clips through LSTM
        clip_tensors = torch.FloatTensor(
            np.stack([c for _, _, c in clips])
        ).to(DEVICE)   # [num_clips, 16, 2048]

        with torch.no_grad():
            scores = self.lstm(clip_tensors).squeeze(1).cpu().numpy()  # [num_clips]

        if self.verbose:
            print(f"  [diag] {len(clips)} clips — "
                  f"scores min={scores.min():.4f}  max={scores.max():.4f}"
                  f"  mean={scores.mean():.4f}  std={scores.std():.4f}")
            n_above = int((scores >= self.threshold).sum())
            print(f"  [diag] clips >= threshold({self.threshold:.2f}) : "
                  f"{n_above}/{len(clips)}")

        return [(s, e, float(sc)) for (s, e, _), sc in zip(clips, scores)]

    def analyze(self, video_path: str) -> VideoResult:
        """
        Run the full pipeline on a video file.

        Args:
            video_path : path to input video (.mp4, .avi, etc.)
        Returns:
            VideoResult with per-clip scores and detections
        """
        t0 = time.time()
        print(f"Analyzing: {Path(video_path).name}")

        # Step 1: Extract frames
        print("  Extracting frames...")
        frames, fps, total_frames = self._extract_frames(video_path)
        duration = len(frames) / fps
        print(f"  {len(frames)} frames @ {fps:.1f}fps ({duration:.1f}s)")

        if len(frames) < CLIP_LENGTH:
            print(f"  WARNING: only {len(frames)} frames extracted "
                  f"(need {CLIP_LENGTH} for one clip). "
                  f"Video too short — returning empty result.")
            return VideoResult(
                video_path      = video_path,
                fps             = fps,
                total_frames    = len(frames),
                duration_sec    = len(frames) / fps if fps > 0 else 0.0,
                clips_analyzed  = 0,
                anomalous_clips = 0,
                max_score       = 0.0,
                results         = [],
                processing_time = time.time() - t0,
            )

        # Step 2: Extract ResNet50 features
        print("  Extracting features (ResNet50)...")
        features = self._frames_to_features(frames)   # [N, 2048]

        # Step 3: Score clips with LSTM
        print("  Scoring clips (LSTM)...")
        clip_scores = self._score_clips(features)     # [(start, end, score), ...]
        print(f"  {len(clip_scores)} clips scored")

        # Step 4: Build results, run YOLO on flagged clips
        results      = []
        anomalous    = []

        for idx, (start, end, score) in enumerate(clip_scores):
            timestamp = start / fps
            is_anom   = score >= self.threshold

            result = ClipResult(
                clip_idx      = idx,
                start_frame   = start,
                end_frame     = end,
                timestamp_sec = timestamp,
                anomaly_score = score,
                is_anomalous  = is_anom,
            )
            results.append(result)
            if is_anom:
                anomalous.append(result)

        # Step 5: Run YOLO + Grad-CAM on the peak frame of each anomalous clip
        if anomalous:
            print(f"  Running YOLO + Grad-CAM on {len(anomalous)} anomalous clips...")
            for result in anomalous:
                # Use the middle frame of the clip as the representative frame
                mid_frame_idx = (result.start_frame + result.end_frame) // 2
                mid_frame_idx = min(mid_frame_idx, len(frames) - 1)
                frame         = frames[mid_frame_idx]

                # YOLO detections
                detections         = self.detector.detect(frame)
                result.detections  = detections
                result.explanation = self.detector.summarize(detections)

                # Grad-CAM heatmap
                heatmap, overlay        = self.gradcam.compute_and_overlay(frame)
                result.heatmap          = heatmap
                result.overlay_image    = overlay

        elapsed = time.time() - t0

        video_result = VideoResult(
            video_path      = video_path,
            fps             = fps,
            total_frames    = len(frames),
            duration_sec    = duration,
            clips_analyzed  = len(clip_scores),
            anomalous_clips = len(anomalous),
            max_score       = max((s for _, _, s in clip_scores), default=0.0),
            results         = results,
            processing_time = elapsed,
        )

        print(f"\n  Done in {elapsed:.1f}s")
        print(f"  Anomalous clips : {len(anomalous)}/{len(clip_scores)}")
        if anomalous:
            print(f"  Peak score      : {video_result.max_score:.4f}")
            print(f"  First anomaly   : {anomalous[0].timestamp_sec:.1f}s")

        return video_result

    def format_report(self, result: VideoResult) -> str:
        """Format a VideoResult as a readable text report."""
        lines = [
            f"=== ANOMALY DETECTION REPORT ===",
            f"Video    : {Path(result.video_path).name}",
            f"Duration : {result.duration_sec:.1f}s  ({result.fps:.1f}fps)",
            f"Clips    : {result.clips_analyzed} analyzed",
            f"Flagged  : {result.anomalous_clips} anomalous clips",
            f"Max score: {result.max_score:.4f}",
            f"",
        ]

        if result.anomalous_clips == 0:
            lines.append("No anomalies detected.")
        else:
            lines.append("ANOMALOUS SEGMENTS:")
            for r in result.results:
                if r.is_anomalous:
                    lines.append(
                        f"  [{r.timestamp_sec:6.1f}s - {r.timestamp_sec + CLIP_LENGTH/result.fps:.1f}s]"
                        f"  score={r.anomaly_score:.4f}  {r.explanation}"
                    )

        lines.append(f"\nProcessing time: {result.processing_time:.1f}s")
        return "\n".join(lines)


def get_pipeline(**kwargs) -> SurveillancePipeline:
    return SurveillancePipeline(**kwargs)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <video_path> [--verbose]")
        print("\nRunning quick component check instead...")

        pipeline = get_pipeline()
        print("\nAll components loaded successfully.")
        print("To analyze a video: python pipeline.py path/to/video.mp4")
        print("To enable diagnostics: python pipeline.py path/to/video.mp4 --verbose")
    else:
        verbose  = "--verbose" in sys.argv
        pipeline = get_pipeline(verbose=verbose)
        result   = pipeline.analyze(sys.argv[1])
        print("\n" + pipeline.format_report(result))
