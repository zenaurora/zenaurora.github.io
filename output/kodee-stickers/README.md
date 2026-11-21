# Kodee 动态表情包

根据提供的角色图制作，共 22 款。保留原图造型，添加手脚摆动、眨眼、跳跃、爱心与泪滴等循环动画。

## 文件怎么用

- `mobile-gif/`：22 个 256 × 256 轻量透明 GIF，优先用于聊天发送。
- `gif/`：22 个 512 × 512 透明 GIF，适合需要较大尺寸的场景。
- `webp/`：22 个透明背景动态 WebP，边缘更平滑，适合网页和支持该格式的应用。
- `png/`：对应的 22 张静态透明 PNG。
- `preview.gif`：6 款精选动作的动态预览。
- `all-stickers.gif`：全部 22 款的动态合集。
- `all-stickers.jpg`：静态索引图。
- `manifest.json`：名称、编号和尺寸等信息。

标准版单张尺寸为 **512 × 512**，轻量 GIF 为 **256 × 256**。每个循环 **2.4 秒、48 帧**，无限循环。GIF 采用单级透明，WebP 和 PNG 保留半透明边缘。素材来自提供的截图，清晰度受原图限制。

## 表情索引

| 编号 | 文件名 | 表情 |
| --- | --- | --- |
| 01 | walking | 走路 |
| 02 | happy | 开心 |
| 03 | greeting | 挥手 |
| 04 | cute | 卖萌 |
| 05 | relaxing | 喝饮料 |
| 06 | running | 奔跑 |
| 07 | running-fast | 冲刺 |
| 08 | jumping | 跳起来 |
| 09 | jumping-playful | 蹦蹦跳跳 |
| 10 | sitting | 坐着晃腿 |
| 11 | joyful | 收到礼物 |
| 12 | sad | 垂头丧气 |
| 13 | crying | 揉眼睛 |
| 14 | holding | 拿着笔 |
| 15 | drawing | 画画中 |
| 16 | in-love | 心动 |
| 17 | in-love-pink | 爱你呀 |
| 18 | angry | 生气 |
| 19 | shocked | 震惊 |
| 20 | dancing | 跳舞 |
| 21 | ice-skates | 滑冰 |
| 22 | heart | 送你一颗心 |

## 继续调整

原图和拆分素材位于 `source/`，制作脚本位于 `tools/`。脚本使用 Python 3、Pillow 和 NumPy，可修改 `animate.py` 中的动作幅度、时长与特效后重新生成。

```sh
python3 -m pip install pillow numpy
python3 tools/prepare.py
python3 tools/animate.py
python3 tools/make_mobile.py
```

GIF、WebP 和 PNG 文件中的角色均不含文字；中文名称仅显示在合集预览里。
