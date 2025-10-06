# s3-vectors-multimodal-playground
Amazon S3 Vectors を使ったマルチモーダル検索の最小サンプルコードです。

## マルチモーダル埋め込みモデル
以下の2つのモデルのサンプルをそれぞれ用意しています。

* Amazon Titan Multimodal Embeddings G1
* Cohere Embed v4

## 検索対象の画像
`/image`配下の画像を利用します。

## 事前準備
### モデルアクセスのリクエスト
使用するモデルを利用開始するには、モデルアクセスのリクエストが必要です。

AWSコンソール Amazon Bedrock > モデルアクセスからリクエストを行ってください。

### S3 Vectors
以下を作成してください。
* S3 Vectorsのバケット
* Amazon Titan Multimodal Embeddings G1 用インデックス
* Cohere Embed v4 用インデックス

## 環境設定
### Python バージョン
Python 3.12 系でのみ動作確認済みです。

### 依存パッケージのインストール
依存関係は [uv](https://github.com/astral-sh/uv) を利用して管理しています。
```
uv sync
```

### .envファイルの設定
`.env.sample`を参考に`.env`ファイルを作成してください。
```
AWS_ACCESS_KEY_ID=YOUR_AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET_ACCESS_KEY
VECTOR_BUCKET_NAME=YOUR_S3_BUCKET_NAME
TITAN_INDEX_NAME=YOUR_TITAN_INDEX_NAME
COHERE_INDEX_NAME=YOUR_COHERE_INDEX_NAME
```

* AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
  * AWS IAM で作成したユーザーのキーを設定してください

* VECTOR_BUCKET_NAME
  * S3 Vectorsのバケット名を設定してください

* TITAN_INDEX_NAME / COHERE_INDEX_NAME
  * それぞれのベクターインデックスの名前を設定してください


## 実行方法
### Amazon Titan Multimodal Embeddings G1
#### 埋め込みベクトルの挿入
```
uv run titan_put_vectors.py
```

#### 画像による類似検索
```
uv run titan_query_vectors_by_image.py
```

#### テキストによる検索
```
uv run titan_query_vectors_by_text.py
```

### Cohere Embed v4
#### 埋め込みベクトルの挿入
```
uv run cohere_put_vectors.py
```

#### 画像による類似検索
```
uv run cohere_query_vectors_by_image.py
```

#### テキストによる検索
```
uv run cohere_query_vectors_by_text.py
```

## 注意
検索対象の画像やテキストは、サンプルコード内に直接指定されています。
必要に応じて編集して利用してください。

## 参考
* [Amazon S3 Vectors](https://aws.amazon.com/s3/features/vectors/)
* [Amazon Titan Multimodal Embeddings](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)
* [Cohere Embed and Cohere Embed v4 models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed.html)

## License
MIT