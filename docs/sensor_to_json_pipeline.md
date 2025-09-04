# センサーデータからJSON構築パイプライン

## ROSトピックから得られる情報

### 座標系の重要な事実
CMU VLAチャレンジでは、**世界座標系が提供されています**：
- `/state_estimation`: ロボットの世界座標系での位置・姿勢
- `/registered_scan`: **すでに世界座標系に変換済み**のLiDARデータ
- 座標変換の複雑さを考える必要はありません！

### 色情報の取得について
**重要：LiDAR点群には色情報が含まれない可能性が高い**
- LiDARは距離センサーであり、通常は`x, y, z, intensity`のみ
- 色情報は`/camera/image`から取得する必要がある
- カメラとLiDARのセンサーフュージョンが必要

## 核心的な問題と解決
ロボットのセンサーは「生データ」しか取得できません：
- **カメラ**: RGB画素値の配列
- **LiDAR**: 世界座標系での3D点群（`/registered_scan`から）

これらから、JSONに必要な以下の情報をどう抽出するか：
```json
{
  "class": "chair",        // どうやって「椅子」と判定？
  "position": [x, y, z],   // 世界座標系での絶対位置
  "color": "blue"          // ピクセル値から「青」をどう判定？
}
```

## 解決策：段階的な情報抽出パイプライン

### Step 1: 物体検出（何があるか？）

#### 使用技術：オープンボキャブラリ物体検出
```python
from transformers import pipeline

# 事前学習済みモデル（数千種類の物体を認識可能）
detector = pipeline("object-detection", model="google/owlvit-base-patch32")

def detect_objects(image):
    # テキストクエリで検出（CMUチャレンジで出現しそうな物体）
    candidate_labels = [
        "chair", "table", "refrigerator", "window", 
        "potted plant", "bed", "sofa", "lamp", "door"
    ]
    
    detections = detector(
        image, 
        candidate_labels=candidate_labels,
        threshold=0.5
    )
    
    # 結果例：
    # [{
    #   "label": "chair",
    #   "score": 0.92,
    #   "box": {"xmin": 120, "ymin": 80, "xmax": 200, "ymax": 180}
    # }]
    
    return detections
```

**なぜこれで「椅子」と分かるのか？**
- モデルは数百万枚の画像で訓練済み
- 椅子の視覚的特徴（背もたれ、座面、脚）を学習している
- 確信度（score）で判定の確かさを表現

### Step 2: 色の判定（何色か？）

#### カメラ画像からの色抽出（LiDARには色情報がないため）
```python
import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_color(image, bbox):
    # バウンディングボックス内の画像を切り出し
    x1, y1, x2, y2 = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
    roi = image[y1:y2, x1:x2]
    
    # HSV色空間に変換（色の判定に適している）
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 方法1: ヒストグラムで最頻色を特定
    def get_dominant_color_simple(hsv_image):
        # Hue（色相）のヒストグラム
        hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hist)
        
        # Hueから色名に変換
        if dominant_hue < 10 or dominant_hue > 160:
            return "red"
        elif 10 <= dominant_hue < 25:
            return "orange"
        elif 25 <= dominant_hue < 35:
            return "yellow"
        elif 35 <= dominant_hue < 85:
            return "green"
        elif 85 <= dominant_hue < 125:
            return "blue"
        elif 125 <= dominant_hue < 145:
            return "purple"
        else:
            return "red"  # ピンク～赤の範囲
    
    # 方法2: KMeansクラスタリングで主要色を抽出
    def get_dominant_color_advanced(roi_bgr):
        pixels = roi_bgr.reshape(-1, 3)
        
        # 3つの主要色にクラスタリング
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(pixels)
        
        # 最も多いクラスタの中心色
        labels = kmeans.labels_
        label_counts = np.bincount(labels)
        dominant_cluster = np.argmax(label_counts)
        dominant_color_bgr = kmeans.cluster_centers_[dominant_cluster]
        
        # BGRからHSVに変換して色名判定
        dominant_hsv = cv2.cvtColor(
            np.uint8([[dominant_color_bgr]]), 
            cv2.COLOR_BGR2HSV
        )[0][0]
        
        return hue_to_color_name(dominant_hsv[0])
    
    return get_dominant_color_simple(hsv)
```

### Step 3: 3D位置の推定（どこにあるか？）

#### ROSトピックを活用した世界座標系での位置推定
```python
import rospy
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2

def estimate_3d_position_world(detection_2d, camera_image):
    """
    2Dバウンディングボックスから世界座標系での3D位置を推定
    CMUが提供する/registered_scanを使用（すでに世界座標系！）
    """
    
    # 世界座標系の点群を取得
    registered_cloud_msg = rospy.wait_for_message("/registered_scan", PointCloud2)
    points_world = []
    for point in pc2.read_points(registered_cloud_msg, field_names=("x", "y", "z")):
        points_world.append([point[0], point[1], point[2]])
    points_world = np.array(points_world)
    
    # カメラの内部パラメータ（キャリブレーション済み）
    camera_matrix = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    
    # ロボットの現在位置を取得（世界座標系）
    robot_odom = rospy.wait_for_message("/state_estimation", Odometry)
    robot_pose = robot_odom.pose.pose
    
    # 世界座標系の点をカメラ座標系に一時的に変換（投影のため）
    def project_world_points_to_image(points_world, robot_pose):
        # ロボット座標系への変換
        points_robot = world_to_robot_transform(points_world, robot_pose)
        
        # カメラ座標系への変換（ロボットに固定されたカメラ）
        points_camera = robot_to_camera_transform(points_robot)
        
        # 2D画像への投影
        points_2d_homo = camera_matrix @ points_camera.T
        points_2d = points_2d_homo[:2] / points_2d_homo[2]  # 正規化
        
        return points_2d.T
    
    # バウンディングボックス内の世界座標系の点を抽出
    bbox = detection_2d["box"]
    points_2d = project_world_points_to_image(points_world, robot_pose)
    
    # bbox内の点のみ選択（世界座標系のまま）
    mask = (
        (points_2d[:, 0] >= bbox["xmin"]) &
        (points_2d[:, 0] <= bbox["xmax"]) &
        (points_2d[:, 1] >= bbox["ymin"]) &
        (points_2d[:, 1] <= bbox["ymax"])
    )
    
    # 世界座標系での物体位置を取得
    roi_points_world = points_world[mask]
    
    if len(roi_points_world) > 0:
        # 世界座標系での物体の中心位置
        position_world = np.mean(roi_points_world, axis=0)
        
        # 位置の不確かさ（世界座標系）
        if len(roi_points_world) > 3:
            covariance = np.cov(roi_points_world.T)
        else:
            covariance = np.eye(3) * 0.1
    else:
        # 点群が少ない場合はカメラのみから推定
        # （実装省略：深度推定モデルを使用）
        position_world = estimate_from_camera_only(bbox, camera_image, robot_pose)
        covariance = np.eye(3) * 1.0  # 大きな不確かさ
    
    return position_world, covariance
```

### Step 4: 統合してJSON構築（世界座標系）

```python
def build_json_from_sensors(camera_image, question):
    """
    センサーデータから世界モデルJSONを構築
    /registered_scanを使用するため、すべて世界座標系で処理
    """
    world_json = {"objects": {}}
    
    # Step 1: 物体検出
    detections = detect_objects(camera_image)
    
    for detection in detections:
        # Step 2: 色抽出
        color = extract_color(camera_image, detection["box"])
        
        # Step 3: 世界座標系での3D位置推定
        position_world, covariance = estimate_3d_position_world(
            detection, camera_image
        )
        
        # Step 4: データアソシエーション（既存物体との照合）
        is_new = True
        for obj_id, existing in world_json["objects"].items():
            if existing["class"] == detection["label"]:
                # マハラノビス距離で同一判定（世界座標系で直接比較可能！）
                if calculate_mahalanobis(position_world, covariance, 
                                        existing["position"], 
                                        existing["position_uncertainty"]) < 9.21:
                    # 既存物体を更新（世界座標系のまま）
                    update_with_ekf(existing, position_world, covariance)
                    is_new = False
                    break
        
        if is_new:
            # 新規物体として追加（世界座標系での絶対位置）
            new_id = str(len(world_json["objects"]))
            world_json["objects"][new_id] = {
                "class": detection["label"],
                "position": position_world.tolist(),  # 世界座標系
                "position_uncertainty": covariance.tolist(),
                "attributes": {
                    "color": color
                }
            }
    
    return world_json
```

## 実装上の重要な判断

### 1. 物体クラスの信頼性
```python
# 確信度が低い場合の処理
if detection["score"] < 0.7:
    # "unknown_object"として記録
    obj["class"] = f"unknown_{detection['label']}"
    obj["confidence"] = detection["score"]
```

### 2. 色判定の曖昧性対処
```python
# 複数の色が混在する場合
if has_multiple_colors(roi):
    obj["attributes"]["color"] = "multicolor"
    obj["attributes"]["color_details"] = ["blue", "white"]
```

### 3. 位置推定の失敗対策
```python
# LiDAR点群が取得できない場合
if len(roi_points_3d) == 0:
    # カメラのみから推定（精度は落ちる）
    obj["position_uncertainty"] = np.eye(3) * 1.0  # 大きな不確かさ
```

## まとめ

### センサーフュージョンによるJSON構築

**各センサーの役割分担：**
1. **カメラ (`/camera/image`)**
   - 物体クラスの検出（"chair", "table"等）
   - 色情報の抽出（"blue", "red"等）

2. **LiDAR (`/registered_scan`)**
   - 世界座標系での3D位置
   - 物体のサイズ・形状
   - 注意：通常、色情報は含まない

3. **統合**
   - カメラの2D検出領域とLiDARの3D点群を対応付け
   - 世界座標系で一貫したJSONを構築

### 実装のポイント

1. **点群フィールドの確認が重要**
   ```python
   # 実行時に点群のフィールドを確認
   for field in cloud_msg.fields:
       print(field.name)  # x, y, z, intensity, rgb?
   ```

2. **色情報の取得方法**
   - 通常：カメラ画像から抽出（本ドキュメントのStep 2）
   - もしRGB付き点群があれば：直接点群から取得可能

3. **世界座標系の利点**
   - ロボットが移動しても物体位置が不変
   - マハラノビス距離によるデータアソシエーションが直接可能

このアプローチにより、探索を通じて一貫性のある世界モデルJSONを構築できます。