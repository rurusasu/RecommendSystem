ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ARG PROJECT_NAME=Kakeibo
ARG USER_NAME=challenger
#ARG GROUP_NAME=challengers
#ARG UID=1000
#ARG GID=1000
ARG APPLICATION_DIRECTORY=/home/${USER_NAME}/${PROJECT_NAME}

# 日本語化
USER root
RUN apt-get update && \
  apt-get install --no-install-recommends -y \
  fonts-noto-cjk \
  fonts-noto-cjk-extra && \
  apt-get install locales -y && \
  # ja_JP.UTF-8の行のコメントを解除
  sed -i -E 's/# (ja_JP.UTF-8)/\1/' /etc/locale.gen && \
  locale-gen && \
  update-locale LANG=ja_JP.UTF-8

# 環境変数の設定
ENV DEBIAN_FRONTEND="noninteractive" \
  TZ="Asia/Tokyo" \
  LC_ALL="ja_JP.UTF-8" \
  LANG="ja_JP.UTF-8" \
  LANGUAGE="ja_JP:ja"

# パッケージインストール
RUN apt-get update && \
  apt-get install --no-install-recommends -y \
  sudo \
  git \
  curl \
  # curl: (77) error setting certificate verify locations への対策
  # ref: https://askubuntu.com/questions/1390288/curl-77-error-setting-certificate-verify-locations-ubuntu-20-04-3-lts
  ca-certificates

# Rye install
# 2024/05/24 追記
# rye のダウンロード先が変更
# ref: [gitリポジトリのReadme](https://github.com/astral-sh/rye)
# 2024/05/24 追記
# RYE_INSTALL_OPTION="--yes" がない場合、インストールエラー
# ref: [Error: IO error: not a terminal #246](https://github.com/astral-sh/rye/issues/246)
RUN /bin/bash -c "curl -sSf https://rye.astral.sh/get | RYE_INSTALL_OPTION='--yes' bash && \
  echo 'source \"$HOME/.rye/env\"' >> $HOME/.bashrc && \
  source $HOME/.rye/env && \
  rye config --set-bool behavior.global-python=true && \
  rye config --set-bool behavior.use-uv=true"

# 不要なパッケージの削除
RUN apt-get autoremove -y && \
  apt-get -y clean && \
  rm -rf /var/lib/apt/lists/*

#USER ${USER_NAME}
#WORKDIR ${APPLICATION_DIRECTORY}
