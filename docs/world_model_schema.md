# 動的世界モデル JSONスキーマ仕様書

## 概要
CMU VLAチャレンジにおいて、ロボットが探索しながら構築する世界モデルのJSON構造を定義します。
本仕様はBlueprintの「質問駆動型」理念に基づき、質問回答に必要な情報のみを効率的に管理します。

## JSON構造定義

```json
{
  "objects": {
    "<object_id>": {
      "class": "string",
      "position": [x, y, z],
      "position_uncertainty": [[3x3 covariance matrix]],
      "attributes": {
        "color": "string"
      }
    }
  }
}
```

### 各フィールドの役割
- **objects**: 検出した物体のコレクション
- **class**: 物体の種類（chair, refrigerator, window等）
- **position**: 3次元空間での物体の位置
- **position_uncertainty**: 位置推定の不確かさ（共分散行列）
- **attributes.color**: 物体の色情報

## 質問タイプ別の必要情報

### 1. 数値クエリ「青い椅子は何個？」

**必要な情報:**
- 物体のclass（椅子かどうか）
- 物体のcolor（青かどうか）

**JSON例:**
```json
{
  "objects": {
    "0": {
      "class": "chair",
      "position": [1.5, 2.3, 0.4],
      "position_uncertainty": [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
      "attributes": {"color": "blue"}
    },
    "1": {
      "class": "chair",
      "position": [3.2, 1.1, 0.4],
      "position_uncertainty": [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
      "attributes": {"color": "blue"}
    }
  }
}
```

**回答生成:**
```python
count = len([obj for obj in objects.values() 
            if obj["class"] == "chair" and obj["attributes"]["color"] == "blue"])
```

### 2. 物体参照「冷蔵庫に最も近い鉢植え」

**必要な情報:**
- 冷蔵庫の位置
- 鉢植えの位置（複数）

**JSON例:**
```json
{
  "objects": {
    "0": {
      "class": "refrigerator",
      "position": [4.0, -2.0, 0.9],
      "position_uncertainty": [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
      "attributes": {}
    },
    "1": {
      "class": "potted_plant",
      "position": [3.5, -1.5, 0.3],
      "position_uncertainty": [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
      "attributes": {}
    },
    "2": {
      "class": "potted_plant",
      "position": [5.0, -3.0, 0.3],
      "position_uncertainty": [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
      "attributes": {}
    }
  }
}
```

**回答生成:**
```python
fridge = next(obj for obj in objects.values() if obj["class"] == "refrigerator")
plants = [obj for obj in objects.values() if obj["class"] == "potted_plant"]
closest = min(plants, key=lambda p: distance(p["position"], fridge["position"]))
```

### 3. 指示追従「窓の近くを通って冷蔵庫へ」

**必要な情報:**
- 窓の位置
- 冷蔵庫の位置

**JSON例:**
```json
{
  "objects": {
    "0": {
      "class": "window",
      "position": [2.0, 0.0, 1.5],
      "position_uncertainty": [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
      "attributes": {}
    },
    "1": {
      "class": "refrigerator",
      "position": [4.0, -2.0, 0.9],
      "position_uncertainty": [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
      "attributes": {}
    }
  }
}
```

**回答生成:**
```python
window = next(obj for obj in objects.values() if obj["class"] == "window")
fridge = next(obj for obj in objects.values() if obj["class"] == "refrigerator")
waypoints = [
    {"x": window["position"][0], "y": window["position"][1], "theta": 0},
    {"x": fridge["position"][0], "y": fridge["position"][1], "theta": 0}
]
```

## データアソシエーション（ID重複防止）

**Blueprintに準拠したマハラノビス距離による判定:**

```python
def is_same_object(new_detection, existing_object):
    """
    マハラノビス距離でID重複を防ぐ（Blueprint準拠）
    """
    # 位置の差分
    diff = np.array(new_detection["position"]) - np.array(existing_object["position"])
    
    # 共分散の和
    cov_sum = (new_detection["position_uncertainty"] + 
              existing_object["position_uncertainty"])
    
    # マハラノビス距離
    mahalanobis = diff.T @ np.linalg.inv(cov_sum) @ diff
    
    # 閾値判定（χ²分布95%信頼区間）
    return mahalanobis < 9.21
```

## 探索中のJSON更新フロー

### Step 1: 質問受信時（空のJSON）
```json
{
  "objects": {}
}
```

### Step 2: ターゲット物体を発見するたび追加
```python
def on_detection(detection):
    # 既存物体との照合
    for obj_id, obj in world_json["objects"].items():
        if obj["class"] == detection["class"]:
            if is_same_object(detection, obj):
                # 既存物体を更新（EKF）
                update_object(obj_id, detection)
                return
    
    # 新規物体として追加
    new_id = str(len(world_json["objects"]))
    world_json["objects"][new_id] = {
        "class": detection["class"],
        "position": detection["position"],
        "position_uncertainty": detection["uncertainty"],
        "attributes": {"color": detection.get("color", "unknown")}
    }
```

### Step 3: 回答生成
質問に必要な情報が揃った時点で、JSONデータから回答を生成します。

## 設計の特徴

### 質問駆動型アプローチ
ロボットは質問内容に基づいて、必要な物体のみを探索・記録します。
例えば「青い椅子は何個？」という質問では、椅子と判定された物体の位置と色情報を重点的に収集します。

### ID重複防止メカニズム
Blueprintで規定されたマハラノビス距離を用いることで、同一物体を複数回観測しても正確に識別できます。
position_uncertaintyフィールドがこの計算を可能にします。

### リアルタイム更新
探索中、新しい物体を発見するたびにJSONが更新されます。
既存物体の再観測時は、EKF（拡張カルマンフィルタ）により位置精度が向上します。

## まとめ

本スキーマは、CMU VLAチャレンジの3つの課題タイプ（数値クエリ、物体参照、指示追従）すべてに対応可能な、
シンプルかつ効率的な構造です。Blueprintのアーキテクチャに準拠し、
質問回答に必要な最小限の情報のみを管理することで、高速かつ正確な処理を実現します。