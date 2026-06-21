# 02 — 软件架构设计

## 1. 总体信号流

```
┌─────────┐     7.1 PCM      ┌──────────────────┐     loopback 8ch     ┌───────────────┐
│  Game   │ ───────────────▶ │ VoiceMeeter 7.1  │ ───────────────────▶ │  Sound Radar  │
│ (7.1模式)│  (HRTF OFF)      │  (虚拟 7.1 设备)  │   (WASAPI loopback)  │     App       │
└─────────┘                  └──────────────────┘                      └───────────────┘
                                      │                                        │
                                      │ 二次 HRTF 虚拟化                        │ 多声道能量分析
                                      ▼                                        ▼
                              ┌────────────────┐                      ┌──────────────┐
                              │  玩家物理耳机   │                      │   雷达 UI    │
                              │(听到方位化音频) │                      │ (实时方位显示)│
                              └────────────────┘                      └──────────────┘
```

**关键点**:雷达拿到的 8 声道信号中,每个声源已按水平方位被分配到对应扬声器,方位信息被物理编码进能量分布。

## 2. 分层结构

软件按职责严格分层,层之间只通过定义好的数据结构通信:

```
┌───────────────────────────────────────────────────┐
│                   UI Layer                         │
│   radar_widget.py  /  main_window.py              │
│   (PyQt6, painter-based radar drawing)            │
└───────────────────────┬───────────────────────────┘
                        │ RadarContact[] (display models)
┌───────────────────────▼───────────────────────────┐
│                Tracking Layer                      │
│   smoother.py  (temporal smoothing, decay)        │
└───────────────────────┬───────────────────────────┘
                        │ DirectionEstimate (event)
┌───────────────────────▼───────────────────────────┐
│                Analysis Layer                     │
│   features.py  (per-channel RMS, band energy)     │
│   noise_floor.py  (adaptive noise estimation)     │
│   onset.py  (spectral-flux event detection)       │
│   direction.py  (VBAP direction estimation)       │
└───────────────────────┬───────────────────────────┘
                        │ AudioFrame (multichannel PCM)
┌───────────────────────▼───────────────────────────┐
│                 Audio Layer                        │
│   capture.py  (WASAPI loopback via PyAudioWPatch)   │
│   framing.py  (ring buffer, windowing)            │
└───────────────────────────────────────────────────┘
```

## 3. 数据流(每帧)

每一帧(`frame_size` 个采样,默认 1024 @48k ≈ 21ms)的处理流水:

| 阶段 | 输入 | 处理 | 输出 |
|---|---|---|---|
| 1. 捕获 | — | WASAPI loopback | `AudioFrame` (shape=`[frame_size, n_channels]`, float32) |
| 2. 特征 | `AudioFrame` | 每声道 RMS^2 + 子带能量 | `FrameFeatures` (8维能量向量 + 频带能量) |
| 3. 本底 | `FrameFeatures` | 滑动最小值 + 时间衰减 | `noise_floor` (标量,float) |
| 4. Onset | `AudioFrame` + 历史频谱 | spectral flux | `OnsetEvent \| None` |
| 5. 方向 | `FrameFeatures` + `noise_floor` | VBAP 插值 | `DirectionEstimate \| None` (angle_deg, confidence) |
| 6. 平滑 | `DirectionEstimate` | 指数衰减 + 角度低通 | `RadarContact[]` |
| 7. UI | `RadarContact[]` | 绘制 | 屏幕 |

## 4. 线程模型(关键设计)

音频回调是**实时线程**,绝不能阻塞、不能做重活、不能触碰 Qt。所以采用 **生产者-消费者 + 队列**解耦:

```
┌──────────────────┐   frame_queue   ┌──────────────────┐
│  Audio Thread    │  (maxsize=4,    │  Processing      │
│  (PortAudio cb)  │ ──────────────▶│  Thread          │
│                  │   drop on full) │  (numpy/scipy)   │
└──────────────────┘                 └────────┬─────────┘
                                              │ contact_queue
                                              ▼
                                     ┌──────────────────┐
                                     │  Qt UI Thread    │
                                     │  (QTimer poll)   │
                                     └──────────────────┘
```

### 防御性规则
- 音频回调只做"复制 PCM 到队列",**不做任何 DSP**
- 队列满时**丢弃旧帧**(实时性 > 完整性)
- UI 线程通过 `QTimer` 定时(60Hz)从 `contact_queue` 取数据
- 禁止跨线程共享可变对象,所有传递的数据都是**不可变快照**

## 5. 配置系统

- 全局配置:`config.yaml`(项目根目录)
- 可选 profile:`profiles/<name>.yaml`(覆盖默认值,但**核心代码不依赖**任何 profile)
- 加载优先级:profile > config.yaml > 硬编码默认值
- 加载即校验:类型、范围、声道布局合法性

详见 `03_module_io.md`。

## 6. 关键数据结构(详见 `03_module_io.md`)

```python
@dataclass
class AudioFrame:
    samples: np.ndarray        # shape=(frame_size, n_channels), float32, [-1,1]
    sample_rate: int
    timestamp: float           # seconds, monotonic

@dataclass
class FrameFeatures:
    channel_energy: np.ndarray # shape=(n_channels,), float, RMS^2, >=0
    band_energy: np.ndarray    # shape=(n_bands,), float, sub-band energy

@dataclass
class DirectionEstimate:
    angle_deg: float           # [-180, 180], 0=front, +x=clockwise
    confidence: float          # [0, 1]
    contributing_channels: list[str]

@dataclass
class RadarContact:
    angle_deg: float
    intensity: float           # [0, 1], decays over time
    born_at: float             # timestamp
```

## 7. 错误处理策略

| 场景 | 处理 |
|---|---|
| WASAPI 设备不存在 | 启动时报错并列出可用设备,引导用户配置 |
| 实际声道数 ≠ 配置声道布局 | 启动时报错,不静默下混 |
| 音频回调异常 | 记录日志,继续运行(不让 UI 崩) |
| UI 线程异常 | 弹窗 + 写日志,不退出进程 |
| 队列持续丢帧(过载) | UI 显示降级警告 |

## 8. 性能预算

- 帧时长:21ms(@48k, 1024 samples)
- 单帧处理目标:< 5ms(numpy 向量化)
- UI 刷新率:60Hz
- CPU 总占用目标:< 10%(单核)

## 9. 可扩展点

- 替换 audio layer 即可适配其他平台(CoreAudio/PulseAudio)
- analysis layer 是纯函数,可单元测试,也可换 ML 模型
- UI layer 与算法解耦,可换 web 前端

## 10. 相关文档

- `03_module_io.md` — 每个模块的 I/O 契约
- `05_algorithm.md` — 算法细节