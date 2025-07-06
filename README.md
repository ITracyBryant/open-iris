## 环境

python3 version = 3.11.13 (<3.12)
```shell
conda create -n open_iris python=3.11
```

## 本地安装执行

```shell
#安装
cd open_iris_tracker
pip install .

#运行
open-iris-tracker
```

## 打包pyz命令:

```shell
python -m zipapp open_iris_tracker -m "main:main" -o open_iris_tracker.pyz
```

