#!/usr/bin/env python3
"""
ROSトピックから画像を取得してOWL-ViTで物体検出
Hugging Face公式の実装方法を使用
"""

import rospy
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge, CvBridgeError
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import torch
import cv2
import numpy as np
import json

class VisionDetector:
    def __init__(self):
        """初期化"""
        # ROSノード初期化
        rospy.init_node('vision_detector', anonymous=True)
        
        # CvBridge初期化（ROS画像↔OpenCV変換）
        self.bridge = CvBridge()
        
        # OWL-ViTモデルとプロセッサをロード（公式方法）
        rospy.loginfo("Loading OWL-ViT model and processor...")
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        
        # GPUが使える場合は使用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # 評価モード
        
        rospy.loginfo(f"Model loaded successfully on {self.device}!")
        
        # 検出対象の物体リスト
        self.target_objects = [
            "chair", "table", "refrigerator", "window",
            "potted plant", "bed", "sofa", "lamp", "door",
            "cabinet", "desk", "television", "microwave"
        ]
        
        # 画像カウンタ（デバッグ用）
        self.image_count = 0
        
        # サブスクライバー設定
        self.image_sub = rospy.Subscriber(
            "/camera/image",
            ROSImage,
            self.image_callback,
            queue_size=1
        )
        
        rospy.loginfo("Vision Detector initialized! Waiting for images...")
    
    def image_callback(self, msg):
        """
        画像を受信したときの処理
        """
        self.image_count += 1
        
        try:
            # ROS ImageメッセージをOpenCV形式（BGR）に変換
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 画像サイズを確認
            height, width = cv_image.shape[:2]
            rospy.loginfo(f"Received image #{self.image_count}: {width}x{height}")
            
            # 360度画像かチェック（幅が高さの2倍以上なら360度画像の可能性）
            is_360 = width >= height * 2
            if is_360:
                rospy.loginfo("Detected 360-degree image")
            
            # 物体検出実行
            detections = self.detect_objects(cv_image)
            
            # 結果を表示
            self.display_results(detections, cv_image)
            
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Error in image callback: {e}")
    
    def detect_objects(self, image):
        """
        OWL-ViTで物体検出（Hugging Face公式方法）
        """
        rospy.loginfo("Running object detection...")
        
        # OpenCV画像をPIL Imageに変換
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # テキストクエリを準備（各物体を"a photo of a [object]"形式に）
        text_queries = [[f"a photo of a {obj}" for obj in self.target_objects]]
        
        # プロセッサで入力を準備
        inputs = self.processor(
            text=text_queries, 
            images=pil_image, 
            return_tensors="pt"
        ).to(self.device)
        
        # 推論実行
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 画像サイズ（高さ、幅）
        target_sizes = torch.Tensor([pil_image.size[::-1]]).to(self.device)
        
        # 後処理（バウンディングボックスとスコアを取得）
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=0.3,  # 信頼度閾値
            target_sizes=target_sizes
        )
        
        # 結果を整理（最初の画像の結果を取得）
        i = 0
        boxes = results[i]["boxes"].cpu().numpy()
        scores = results[i]["scores"].cpu().numpy()
        labels = results[i]["labels"].cpu().numpy()
        
        rospy.loginfo(f"Detected {len(boxes)} objects")
        
        # 検出結果をフォーマット
        detections = []
        for box, score, label_idx in zip(boxes, scores, labels):
            # ラベルインデックスから実際のラベル名を取得
            label_name = self.target_objects[label_idx]
            
            detection = {
                "label": label_name,
                "score": float(score),
                "box": {
                    "xmin": int(box[0]),
                    "ymin": int(box[1]),
                    "xmax": int(box[2]),
                    "ymax": int(box[3])
                }
            }
            detections.append(detection)
            
            # ログ出力
            rospy.loginfo(
                f"  - {detection['label']}: "
                f"score={detection['score']:.3f}, "
                f"box=[{detection['box']['xmin']}, {detection['box']['ymin']}, "
                f"{detection['box']['xmax']}, {detection['box']['ymax']}]"
            )
        
        return detections
    
    def display_results(self, detections, image):
        """
        検出結果を画像に描画して表示（デバッグ用）
        """
        # 画像をコピー（元画像を変更しないため）
        display_image = image.copy()
        
        # 各検出結果を描画
        for det in detections:
            # バウンディングボックス
            cv2.rectangle(
                display_image,
                (det["box"]["xmin"], det["box"]["ymin"]),
                (det["box"]["xmax"], det["box"]["ymax"]),
                (0, 255, 0),  # 緑色
                2
            )
            
            # ラベルとスコア
            label_text = f"{det['label']}: {det['score']:.2f}"
            cv2.putText(
                display_image,
                label_text,
                (det["box"]["xmin"], det["box"]["ymin"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # ウィンドウに表示（オプション）
        # 注意：SSHやDockerでは表示できない場合があります
        try:
            cv2.imshow("Detections", display_image)
            cv2.waitKey(1)  # 1msだけ待機（画面更新のため）
        except:
            pass  # 表示できない環境では無視
        
        # 画像を保存（デバッグ用）
        if self.image_count % 10 == 0:  # 10フレームごとに保存
            filename = f"/tmp/detection_{self.image_count:06d}.jpg"
            cv2.imwrite(filename, display_image)
            rospy.loginfo(f"Saved detection result to {filename}")
    
    def spin(self):
        """
        メインループ
        """
        rospy.spin()

def main():
    """
    メイン関数
    """
    try:
        detector = VisionDetector()
        detector.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Vision Detector terminated")
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt, shutting down")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()