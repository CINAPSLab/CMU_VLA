# CMU VLA ROS Topics Documentation

このドキュメントは、CMU Vision-Language-Autonomy Challengeシステムで利用可能なROSトピックの説明です。

## AIモジュールインターフェーストピック

### 入力トピック（AIモジュールへ）
- `/challenge_question` (std_msgs/String): 自然言語クエリ
- `/camera/image` (sensor_msgs/Image): 360°カメラフィード
- `/registered_scan` (sensor_msgs/PointCloud2): 登録済み3Dライダーデータ
- `/sensor_scan` (sensor_msgs/PointCloud2): 生のライダーデータ
- `/terrain_map` (sensor_msgs/PointCloud2): ローカル地形解析データ
- `/state_estimation` (nav_msgs/Odometry): 車両姿勢
- `/traversable_area` (sensor_msgs/PointCloud2): ナビゲーション可能エリアマップ
- `/object_markers` (visualization_msgs/MarkerArray): グラウンドトゥルース意味情報（学習時のみ）

### 出力トピック（AIモジュールから）
- `/numerical_response` (std_msgs/Int32): 「いくつ」質問用の数値回答
- `/selected_object_marker` (visualization_msgs/Marker): 「～を見つけて」質問用のオブジェクト回答
- `/way_point_with_heading` (geometry_msgs/Pose2D): ナビゲーション指示用のウェイポイント

## ナビゲーション・制御トピック
- `/cmd_vel` (geometry_msgs/Twist): 速度コマンド
- `/path` (nav_msgs/Path): 計画されたパス
- `/trajectory` (nav_msgs/Path): 車両軌道
- `/navigation_boundary` (sensor_msgs/PointCloud2): ナビゲーション境界
- `/free_paths` (sensor_msgs/PointCloud2): 利用可能なナビゲーションパス
- `/way_point` (geometry_msgs/PointStamped): ナビゲーションウェイポイント
- `/speed` (std_msgs/Float32): 現在の速度
- `/stop` (std_msgs/Bool): ストップコマンド

## 知覚・マッピングトピック
- `/camera/depth_image` (sensor_msgs/Image): 深度カメラデータ
- `/camera/depth_image/compressed` (sensor_msgs/CompressedImage): 圧縮深度画像
- `/camera/semantic_image` (sensor_msgs/Image): セマンティック画像
- `/camera/semantic_image/compressed` (sensor_msgs/CompressedImage): 圧縮セマンティック画像
- `/semantic_scan` (sensor_msgs/PointCloud2): セマンティック点群
- `/overall_map` (sensor_msgs/PointCloud2): 全体環境マップ
- `/explored_areas` (sensor_msgs/PointCloud2): マップ済み領域
- `/explored_volume` (sensor_msgs/PointCloud2): 3D探索済み空間

## 障害物検知トピック
- `/added_obstacles` (sensor_msgs/PointCloud2): 追加された障害物
- `/check_obstacle` (sensor_msgs/PointCloud2): 障害物チェック用データ

## ユーティリティトピック
- `/joy` (sensor_msgs/Joy): ジョイスティック入力
- `/joy/set_feedback` (sensor_msgs/JoyFeedback): ジョイスティックフィードバック
- `/map_clearing` (sensor_msgs/PointCloud2): マップクリアリング
- `/cloud_clearing` (sensor_msgs/PointCloud2): 点群クリアリング

## システム監視トピック
- `/diagnostics` (diagnostic_msgs/DiagnosticArray): システムヘルス監視
- `/diagnostics_agg` (diagnostic_msgs/DiagnosticArray): 統合診断情報
- `/diagnostics_toplevel_state` (diagnostic_msgs/DiagnosticStatus): トップレベル状態
- `/rosout` (rosgraph_msgs/Log): ROSログ
- `/rosout_agg` (rosgraph_msgs/Log): 統合ROSログ
- `/runtime` (std_msgs/Float32): システムランタイム情報
- `/time_duration` (std_msgs/Duration): タイミング情報
- `/traveling_distance` (std_msgs/Float32): 移動距離

## Transform・座標系トピック
- `/tf` (tf2_msgs/TFMessage): リアルタイム座標変換
- `/tf_static` (tf2_msgs/TFMessage): 静的座標変換

## Unity連携トピック
- `/unity_sim/set_model_state` (gazebo_msgs/SetModelState): Unityシミュレータモデル状態設定

## 画像圧縮パラメータトピック
- `/republish_depth_image/compressed/parameter_descriptions`
- `/republish_depth_image/compressed/parameter_updates`
- `/republish_image/compressed/parameter_descriptions`
- `/republish_image/compressed/parameter_updates`
- `/republish_sem_image/compressed/parameter_descriptions`
- `/republish_sem_image/compressed/parameter_updates`

## 質問タイプ別処理

### 1. 数値質問（Numerical Questions）
条件に合致するオブジェクトの数をカウントして整数で回答
- 入力: `/challenge_question`
- 出力: `/numerical_response`

### 2. オブジェクト参照質問（Object Reference Questions）
特定のオブジェクトを見つけてバウンディングボックスマーカーで回答
- 入力: `/challenge_question`
- 出力: `/selected_object_marker`

### 3. 指示追従質問（Instruction-Following Questions）
制約条件付きでパスをナビゲートし、ウェイポイントシーケンスで回答
- 入力: `/challenge_question`
- 出力: `/way_point_with_heading`

## 注意事項
- チャレンジモードでは10分の制限時間があります
- 各質問に対してシステムが再起動されるため、状態は保持されません
- 学習時のみ`/object_markers`トピックが利用可能です