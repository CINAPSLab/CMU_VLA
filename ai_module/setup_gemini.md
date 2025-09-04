# Gemini API セットアップガイド

## 1. Google AI Studio でAPIキーを取得

1. [Google AI Studio](https://aistudio.google.com/) にアクセス
2. Googleアカウントでログイン
3. 左上の「Get API key」をクリック
4. 「APIキーを作成」をクリック
5. プロジェクトを選択（初回は「Gemini API」を選択）
6. APIキーをコピーして安全に保管

## 2. 環境変数の設定

### Linux/Mac
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 永続的に設定する場合
```bash
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## 3. 必要なPythonパッケージのインストール

```bash
pip install google-generativeai opencv-python
```

## 4. 使用方法

```bash
# ROSワークスペースでビルド
cd ai_module/
catkin_make

# 実行
rosrun dummy_vlm gemini_vlm_example.py
```

## 無料枠の制限

**Gemini 2.0 Flash（推奨）**
- 1分あたり15リクエスト
- 1分あたり100万トークン
- 1日あたり200リクエスト

開発・テスト段階では十分な容量です。

## 注意事項

- 無料版では入力データがGoogleの学習に使用される可能性があります
- 機密情報は送信しないでください
- APIキーは公開リポジトリにコミットしないよう注意してください