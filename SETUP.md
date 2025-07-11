# CMU-VLA-Challenge クイックスタート & 概要

## このコンテストについて
- **CMU Vision-Language-Autonomy Challenge**は、ロボットが自然言語の指示でナビゲーションや物体認識を行うAIの性能を競う国際コンテストです。
- Unityベースのシミュレーション環境や実機ロボットで、与えられた「質問」や「指示」にAIがどれだけ正確に応答できるかを評価します。
- 主なタスクは以下の3種類：
  1. **数値質問**：例「青い椅子は何脚ありますか？」
  2. **オブジェクト参照**：例「冷蔵庫に一番近い鉢植えを見つけて」
  3. **指示追従**：例「窓の近くを通って冷蔵庫まで行って」
- AIモジュールはROSトピック経由でシステムとやりとりします。

---

## WSL+Dockerでの起動方法（クイックガイド）

### 1. 事前準備
- Windowsに**WSL2（Ubuntu 20.04/22.04）**と**Docker Desktop**をインストール
- Windowsで**VcXsrv（XLaunch）**などのXサーバーをインストール
- Docker Desktopの「Resources > WSL Integration」でUbuntuを有効化

### 2. プロジェクトのクローン
```bash
cd ~
git clone https://github.com/CMU-VLA-Challenge/CMU-VLA-Challenge.git
cd CMU-VLA-Challenge/docker
```

### 3. Dockerコンテナの起動
- GPUあり：
  ```bash
  docker compose -f compose_gpu.yml up --build -d
  ```
- GPUなし：
  ```bash
  docker compose -f compose.yml up --build -d
  ```

### 4. コンテナに入る
```bash
docker exec -it ubuntu20_ros_system bash
# 別ターミナルで
docker exec -it ubuntu20_ros bash
```

### 5. システム・AIモジュールのビルド
- システム側：
  ```bash
  cd system/unity
  catkin_make
  ```
- AIモジュール側：
  ```bash
  cd ai_module
  catkin_make
  ```

### 6. Xサーバーの起動とDISPLAY設定
- WindowsでVcXsrv（XLaunch）を「Disable access control」付きで起動
- Docker/WSL内で：
  ```bash
  export DISPLAY=WindowsのIPアドレス:0
  export QT_X11_NO_MITSHM=1
  ```

### 7. システム・AIモジュールの起動
- システム側：
  ```bash
  ./launch_system.sh
  ```
- AIモジュール側：
  ```bash
  ./launch_module.sh
  ```

### 8. RVizやxeyesでGUI確認
- 例：
  ```bash
  xeyes
  rviz
  ```

---

## よくあるトラブル
- **GUIが表示されない**：Xサーバーの起動・DISPLAY設定・ファイアウォールを再確認
- **Permission denied**：`chmod +x`で実行権限を付与
- **ROS masterに接続できない**：同じコンテナ・同じ環境変数でrvizを起動

---

## 参考
- README.md, README_ja.md も参照
- 詳細な質問やエラーはエラーメッセージとともに相談してください 