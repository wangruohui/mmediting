<div align="center">
  <img src="resources/mmediting-logo.png" width="500px"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://badge.fury.io/py/mmedit.svg)](https://pypi.org/project/mmedit/)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmediting.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmediting/workflows/build/badge.svg)](https://github.com/open-mmlab/mmediting/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmediting/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmediting)
[![license](https://img.shields.io/github/license/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmediting.svg)](https://github.com/open-mmlab/mmediting/issues)

[📘使用文档](https://mmediting.readthedocs.io/zh_CN/latest/) |
[🛠️安装教程](https://mmediting.readthedocs.io/zh_CN/latest/install.html) |
[👀模型库](https://mmediting.readthedocs.io/zh_CN/latest/modelzoo.html) |
[🆕更新记录](https://github.com/open-mmlab/mmediting/blob/master/docs/zh_cn/changelog.md) |
[🚀进行中的项目](https://github.com/open-mmlab/mmediting/projects) |
[🤔提出问题](https://github.com/open-mmlab/mmediting/issues)

</div>

[English](/README.md) | 简体中文

## Introduction

MMEditing 是基于 PyTorch 的图像&视频编辑开源工具箱。是 [OpenMMLab](https://openmmlab.com/) 项目的成员之一。

目前 MMEditing 支持下列任务：

<div align="center">
  <img src="https://user-images.githubusercontent.com/12756472/158984079-c4754015-c1f6-48c5-ac46-62e79448c372.jpg"/>
</div>

主分支代码目前支持 **PyTorch 1.5 以上**的版本。

一些示例:

https://user-images.githubusercontent.com/12756472/158972852-be5849aa-846b-41a8-8687-da5dee968ac7.mp4

https://user-images.githubusercontent.com/12756472/158972813-d8d0f19c-f49c-4618-9967-52652726ef19.mp4

### 主要特性

- **模块化设计**

  MMEditing 将编辑框架分解为不同的组件，并且可以通过组合不同的模块轻松地构建自定义的编辑器模型。

- **支持多种编辑任务**

  MMEditing 支持*修复*、*抠图*、*超分辨率*、*生成*等多种主流编辑任务。

- **SOTA**

  MMEditing 提供修复/抠图/超分辨率/生成等任务最先进的算法。

需要注意的是 **MMSR** 已作为 MMEditing 的一部分并入本仓库。
MMEditing 缜密地设计新的框架并将其精心实现，希望能够为您带来更好的体验。

## 最新消息

- \[2022-06-01\] v0.15.0 版本发布
  - 支持 FLAVR
  - 支持 AOT-GAN
  - 新版 CAIN，支持 ReduceLROnPlateau 策略
- \[2022-04-01\] v0.14.0 版本发布
  - 支持视频插帧算法 TOFlow
- \[2022-03-01\] v0.13.0 版本发布
  - 支持 CAIN
  - 支持 EDVR-L
  - 支持在 Windows 系统中运行
- \[2022-02-11\] 切换到 **PyTorch 1.5+**. 将不再保证与早期版本的 PyTorch 的兼容性

请查看 [changelog.md](docs/en/changelog.md) 以获取更多细节与发版记录

## 安装

MMEditing 依赖 [PyTorch](https://pytorch.org/) 和 [MMCV](https://github.com/open-mmlab/mmcv)，以下是安装的简要步骤。

**步骤 1.**
依照[官方教程](https://pytorch.org/get-started/locally/)安装PyTorch

**步骤 2.**
使用 [MIM](https://github.com/open-mmlab/mim) 安装 MMCV

```
pip3 install openmim
mim install mmcv-full
```

**步骤 3.**
从源码安装 MMEditing

```
git clone https://github.com/open-mmlab/mmediting.git
cd mmediting
pip3 install -e .
```

更详细的安装指南请参考 [install.md](../../wiki/1.-Installation) 。

## 开始使用

请参考[使用教程](docs/zh_cn/getting_started.md)和[功能演示](docs/zh_cn/demo.md)获取MMEditing的基本用法。

## 模型库

支持的算法:

<details open>
<summary>图像修复</summary>

- [x] [Global&Local](configs/inpainting/global_local/README.md) (ToG'2017)
- [x] [DeepFillv1](configs/inpainting/deepfillv1/README.md) (CVPR'2018)
- [x] [PConv](configs/inpainting/partial_conv/README.md) (ECCV'2018)
- [x] [DeepFillv2](configs/inpainting/deepfillv2/README.md) (CVPR'2019)
- [x] [AOT-GAN](configs/inpainting/AOT-GAN/README.md) (TVCG'2021)

</details>

<details open>
<summary>图像抠图</summary>

- [x] [DIM](configs/mattors/dim/README.md) (CVPR'2017)
- [x] [IndexNet](configs/mattors/indexnet/README.md) (ICCV'2019)
- [x] [GCA](configs/mattors/gca/README.md) (AAAI'2020)

</details>

<details open>
<summary>图像超分辨率</summary>

- [x] [SRCNN](configs/restorers/srcnn/README.md) (TPAMI'2015)
- [x] [SRResNet&SRGAN](configs/restorers/srresnet_srgan/README.md) (CVPR'2016)
- [x] [EDSR](configs/restorers/edsr/README.md) (CVPR'2017)
- [x] [ESRGAN](configs/restorers/esrgan/README.md) (ECCV'2018)
- [x] [RDN](configs/restorers/rdn/README.md) (CVPR'2018)
- [x] [DIC](configs/restorers/dic/README.md) (CVPR'2020)
- [x] [TTSR](configs/restorers/ttsr/README.md) (CVPR'2020)
- [x] [GLEAN](configs/restorers/glean/README.md) (CVPR'2021)
- [x] [LIIF](configs/restorers/liif/README.md) (CVPR'2021)

</details>

<details open>
<summary>视频超分辨率</summary>

- [x] [EDVR](configs/restorers/edvr/README.md) (CVPR'2019)
- [x] [TOF](configs/restorers/tof/README.md) (IJCV'2019)
- [x] [TDAN](configs/restorers/tdan/README.md) (CVPR'2020)
- [x] [BasicVSR](configs/restorers/basicvsr/README.md) (CVPR'2021)
- [x] [IconVSR](configs/restorers/iconvsr/README.md) (CVPR'2021)
- [x] [BasicVSR++](configs/restorers/basicvsr_plusplus/README.md) (CVPR'2022)
- [x] [RealBasicVSR](configs/restorers/real_basicvsr/README.md) (CVPR'2022)

</details>

<details open>
<summary>图像生成</summary>

- [x] [CycleGAN](configs/synthesizers/cyclegan/README.md) (ICCV'2017)
- [x] [pix2pix](configs/synthesizers/pix2pix/README.md) (CVPR'2017)

</details>

<details open>
<summary>视频插帧</summary>

- [x] [TOFlow](configs/video_interpolators/tof/README.md) (IJCV'2019)
- [x] [CAIN](configs/video_interpolators/cain/README.md) (AAAI'2020)
- [x] [FLAVR](configs/video_interpolators/flavr/README.md) (CVPR'2021)

</details>

请参考[模型库](https://mmediting.readthedocs.io/en/latest/modelzoo.html)了解详情。

## 参与贡献

感谢您为改善 MMEditing 所做的所有贡献。请参阅 MMCV 中的 [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md) 以获取贡献指南。

## 致谢

MMEditing 是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## 引用

如果 MMEditing 对您的研究有所帮助，请按照如下 bibtex 引用它。

```bibtex
@misc{mmediting2022,
    title = {{MMEditing}: {OpenMMLab} Image and Video Editing Toolbox},
    author = {{MMEditing Contributors}},
    howpublished = {\url{https://github.com/open-mmlab/mmediting}},
    year = {2022}
}
```

## 许可证

本项目开源自 [Apache 2.0 license](LICENSE)。

## 学习资料

- [![底层视觉与 MMEditing（上）](https://i2.hdslb.com/bfs/archive/01d51d14a091e96f8c42031390f08f62cb18b699.png@240w_140h_1c.webp)](https://www.bilibili.com/video/BV1zq4y1o7ph/)
  [![底层视觉与 MMEditing（下）](https://i0.hdslb.com/bfs/archive/3c314ffc38bced8002162319cb75f883b4694445.png@240w_140h_1c.webp)](https://www.bilibili.com/video/BV1cQ4y167KL/)

- [零基础 Pytorch 入门超分辨率](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247484944&idx=1&sn=a5beb51cc709484519e3c60c2f9c5557&chksm=ebb50ef2dcc287e42c9da13c784e60a3ca03affdb5313e69c4d7fc6009bc6e45ec34f1275b15&token=1125533908&lang=zh_CN#rd)

- [视觉底层任务优秀开源工作：MMEditing 库使用方法](http://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247489786&idx=1&sn=a103a893b38f66759b969590a98f1475&chksm=ebb51018dcc2990e66ec61b34cc925b87858d3dec41075d71772f0602406e912cd64d07e40b7#rd)

- [让 GLEAN 还原你女神的美妙容颜](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247485438&idx=1&sn=4db5342eee135b66757638743d2b4c5f&chksm=ebb50f1cdcc2860a2e3bce2c7bf4619172959475d4fcf2c39e4662286e8d750b76e93d236f24&token=1125533908&lang=zh_CN#rd)

- [MMEditing | 新视频超分算法冠军BasicVSR++来了](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247483847&idx=1&sn=4c7b166efab7cc15af1f3b12ccd0a7b6&chksm=ebb50925dcc28033f56d3111533ab9fdfd551d6f351dd2a89f130208ed99134ccc3cb65612c4&token=1125533908&lang=zh_CN#rd)

- [不容错过！作者亲自解读 RealBasicVSR](http://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247489508&idx=1&sn=07d27e99713d052c8a088553707f1adf&chksm=ebb51f06dcc29610761f2623fe3e86f0403bbf1c939eb86cce2b55faf3b315ac0d39d39254ac#rd)

- [手把手带你训练 CVPR2022 视频超分模型](http://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247490241&idx=1&sn=519312b9df321181385e84d996c91832&chksm=ebb51223dcc29b35e079408e76d0a58a9486b012c21fa81b8813296c89cf7945ad7589a4821f#rd)

- [一键慢镜头：视频插帧，让老电影“纵享丝滑”](http://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247487939&idx=1&sn=3075c73123e514ca8c3ae4dd903d6d7f&chksm=ebb51921dcc290375224adc5f28a53038112cc5f528475b953014825d32d4f7989277c3260df#rd)

- [还在眨眼补帧？手把手教你插帧算法，让视频顺滑如丝](https://mp.weixin.qq.com/s?__biz=MzI4MDcxNTY2MQ==&mid=2247493512&idx=1&sn=92f81a0ab8b45a686d1100e3aae78144&chksm=ebb6ef6adcc1667c491ec104db9e39257656aae5ffb23fc425228ae5368ab0ebad1d31d0ed22&token=1125533908&lang=zh_CN#rd)

[🧐访问课程主页了解更多🧐](https://github.com/open-mmlab/OpenMMLabCourse)

## OpenMMLab 的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具箱
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=GJP18SjI)，或通过群主小喵加入微信官方交流群。

<div align="center">
<img src="docs/zh_cn/_static/image/zhihu_qrcode.jpg" height="500" />  <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/qq_group_qrcode.jpg" height="500" /> <img src="https://raw.githubusercontent.com/open-mmlab/mmcv/master/docs/en/_static/wechat_qrcode.jpg" height="500" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
