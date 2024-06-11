# RecommendSystem
レコメンドシステムについて勉強するためのリポジトリ

# .env ファイル

`docker-compose.yml` ファイルとその中で使用する環境変数を定義した`.env` ファイルは同じ階層に保存してください。

## .env ファイルの例

```env
CONTAINER_NAME=pgsql_db // コンテナ名
HOST_NAME=pgsql_db // ホスト名
POSTGRES_USER=postgres // ユーザー名
POSTGRES_PASSWORD=postgres // パスワード
DB_NAME=pgsql // データベース名
PGADMIN_DEFAULT_EMAIL=***@gmail.com // pgAdmin4のログインID
PGADMIN_DEFAULT_PASSWORD=123456 // pgAdmin4のログインパスワード
```

## docker-compose 環境変数の確認

`docker-compose.yml` ファイルの環境変数を確認するには、以下のコマンドを実行してください。

```bash
$ docker-compose config