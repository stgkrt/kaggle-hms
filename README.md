# my kaggle template
ぼくの かんがえた さいきょうの かぐるかんきょう

devcontainerでコンテナビルドが終わったときにscripts/setup_dev.shが実行されるようにしています。
(precommitが入るようになっています)
sshから接続した状態でgit cloneから始めると、dotfilesがないとエラーが出ますがリビルドすれば正常に使用できます。
pipのresolverでerrorが出ますがprecommitは動くのでとりあえずそのままで。

# directory
    ├── .devcontainer            <- Container settings.
    ├── input/                   <- Competition Datasets.
    ├── notebooks/               <- Jupyter notebooks.
    ├── scripts/                 <- Scripts.
    ├── src/                     <- Source code. This sould be Python module.
    ├── working/                 <- Output models and train logs.
    │
    ├── .dockerignore
    ├── .gitignore
    ├── .pre-commit-config.yaml  <- pre-commit settings.
    ├── setup.cfg                <- formatter/linter settings in vscode.
    └── README.md                <- The top-level README for developers.
