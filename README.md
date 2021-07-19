# FaceDetection (FasterRCNN)
暑期CV课程的大作业，即便上就是复现一个FasterRCNN。

最终目标为完成一个静态图的多目标人脸识别模型并在`./test.png`上完成多目标人脸检测。

## 项目结构

- `data`(*folder*) : 数据以及data loader的类与函数
- `face_detection`(*folder*) : 实现FasterRCNN的包
- `test`(*folder*) : 测试文件
- `model_test.ipynb`(*file*) : 测试最终的图片
- `unit_test.py`(*file*) : 单元测试
- `train_utils.py`(*file*) : 训练使用的组件
- `visdom_utils.py`(*file*) : visdom前端可视化组件
- `test.ipynb`(*file*) : 评估用的jupyter
- `count.py`(*file*) : 无聊地用来统计项目工程量的item

> 部分程序需通过命令行启动，因为我使用了`fire.Fire()`进行了接口暴露，不熟悉的fire的朋友们可以看看下面这篇blog <[python fire使用指南](https://blog.csdn.net/qq_17550379/article/details/79943740)>。 最终的展示与训练都是在jupyter notebook中进行的

## 开始

请先安装依赖项：
```bash
$pip install -r .\requirements.txt
```

## Utils

需要注意的是，由于我们只做人脸检测而不做人脸识别，所以FasterRCNN中的多目标分类的标签永远是0（代表第一个类，也是唯一一个前景类，也就是人脸）

使用`count.py`统计项目行数的示例如下：

```bash
$python -u .\count.py --path "." --ignore "['2002', '2003', 'FDDB-folds']"
```

> `fire.Fire()`通过`--arg value`的形式传递参数，如果你需要传递`list`，那么请将list用""包起来。

```
.\count.py      94
.\data\data.py  93
.\data\meta.json        36718

......

.\train_utils.py        249
.\unit_test.py  94
.\visdom_utils.py       7
+-----+--------+------+-------+
| CPP | PYTHON | JAVA | TOTAL |
+-----+--------+------+-------+
|  0  |  1543  |  0   |  1543 |
+-----+--------+------+-------+
```