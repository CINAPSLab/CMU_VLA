#!/usr/bin/env python3
"""
Open-Vocabulary Detector wrapper (OWL-ViT).

Safe skeleton: if transformers are not available, returns empty detections.
"""

from typing import List, Dict, Optional


class OwlVitDetector:
    def __init__(self, model_name: str = "google/owlvit-base-patch32", device: Optional[str] = None):
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection  # type: ignore
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "transformers/torch not available; install to enable OwlVitDetector"
            ) from e

        self.OwlViTProcessor = OwlViTProcessor
        self.OwlViTForObjectDetection = OwlViTForObjectDetection
        self.torch = torch

        self.processor = self.OwlViTProcessor.from_pretrained(model_name)
        self.model = self.OwlViTForObjectDetection.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def detect(self, bgr_image, target_objects: List[str], threshold: float = 0.3) -> List[Dict]:
        """Run OWL-ViT detection.

        Args:
          bgr_image: numpy array (H,W,3) in BGR (as cv2 captures)
          target_objects: label strings (will be converted to prompts)
          threshold: score threshold
        Returns:
          List of dicts with keys: label, score, box{xmin,ymin,xmax,ymax}
        """
        if bgr_image is None:
            return []

        import cv2
        from PIL import Image

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # Default prompts if empty
        if not target_objects:
            # Challenge-oriented vocabulary (frequent in questions)
            target_objects = [
                # furniture
                "sofa", "couch", "chair", "stool", "bench", "bed",
                "table", "coffee table", "dining table", "nightstand", "bedside table",
                "cabinet", "tv cabinet", "file cabinet", "shelf", "bookcase",
                # appliances / electronics
                "tv", "television", "microwave", "refrigerator", "computer monitor",
                # decor / small objects
                "pillow", "cushion", "lamp", "lantern", "vase", "bowl", "cup",
                "picture", "painting", "photo", "clock", "guitar", "flowers", "flower",
                "beer bottle", "bottle", "record",
                # structures
                "window", "door", "door frame", "whiteboard", "fireplace",
                # containers / misc
                "trash can", "bin", "box", "folder", "phone",
                # plants
                "potted plant", "plant",
            ]

        text_queries = [[f"a photo of a {obj}" for obj in target_objects]]

        inputs = self.processor(text=text_queries, images=pil_image, return_tensors="pt").to(self.device)

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = self.torch.Tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(outputs=outputs, threshold=threshold, target_sizes=target_sizes)

        i = 0
        boxes = results[i]["boxes"].cpu().numpy()
        scores = results[i]["scores"].cpu().numpy()
        labels = results[i]["labels"].cpu().numpy()

        detections: List[Dict] = []
        for box, score, label_idx in zip(boxes, scores, labels):
            det = {
                "label": target_objects[int(label_idx)] if 0 <= int(label_idx) < len(target_objects) else "object",
                "score": float(score),
                "box": {
                    "xmin": int(box[0]),
                    "ymin": int(box[1]),
                    "xmax": int(box[2]),
                    "ymax": int(box[3]),
                },
            }
            detections.append(det)
        return detections
