# Smart VLM - Vision Detector

ROSトピックから画像を取得してOWL-ViTで物体検出を行うモジュール

## セットアップ

### 1. 依存関係のインストール
```bash
# Python依存関係
pip install -r requirements.txt

# または個別にインストール
pip install transformers torch opencv-python
```

### 2. ビルド
```bash
cd ~/CMU_VLA/ai_module
catkin_make
source devel/setup.bash
```

## 実行方法

### 方法1: Pythonスクリプト直接実行
```bash
# ターミナル1: システムを起動
./launch_system.sh

# ターミナル2: Vision Detectorを実行
cd ~/CMU_VLA/ai_module/src/smart_vlm
python3 vision_detector.py
```

### 方法2: ROSlaunchを使用
```bash
roslaunch smart_vlm vision_detector.launch
```

## 動作確認

1. **ログを確認**
   - 「Model loaded successfully!」が表示されればモデル読み込み成功
   - 「Received image」が表示されれば画像受信成功
   - 「Detected N objects」で検出数を確認

2. **デバッグ画像**
   - `/tmp/detection_*.jpg`に検出結果が保存される（10フレームごと）

## トラブルシューティング

### モデルのダウンロードが遅い
初回実行時はモデル（~600MB）のダウンロードに時間がかかります。

### CUDA/GPUエラー
```python
# CPUで実行する場合
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### メモリ不足
検出対象を減らすか、画像サイズを縮小してください。

## 次のステップ

1. **LiDARとの統合**
   - `/registered_scan`から3D位置を取得
   - 2Dバウンディングボックスと対応付け

2. **世界モデル（JSON）構築**
   - 検出結果をJSONに記録
   - EKFで位置を更新

3. **色情報の抽出**
   - バウンディングボックス内から主要色を判定