# 03 — 模块/方法 输入输出契约

本文档逐模块定义**输入数据类型、范围、单位**与**输出数据类型、范围、含义**。所有实现必须严格遵守此契约。Python 3.9 兼容(类型注解用 `typing.Optional`/`typing.List`,不使用 `X | Y` 运行时语法)。

## 0. 共享数据结构(`src/types.py`)

```python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class AudioFrame:
    """One block of multichannel PCM captured from loopback."""
    samples: np.ndarray          # shape=(frame_size, n_channels), dtype=float32, range [-1, 1]
    sample_rate: int             # e.g. 48000
    timestamp: float             # monotonic seconds (time.monotonic())
    channel_names: List[str]     # e.g. ["L","R","C","LFE","Ls","Rs","Lb","Rb"]

@dataclass
class FrameFeatures:
    """Per-frame acoustic features."""
    channel_energy: np.ndarray   # shape=(n_channels,), dtype=float64, RMS^2, range [0, +inf)
    band_energy: np.ndarray      # shape=(n_bands,),    dtype=float64, range [0, +inf)
    timestamp: float

@dataclass
class DirectionEstimate:
    """Output of direction estimation for a single event."""
    angle_deg: float             # range [-180, 180], 0=front, + = clockwise, - = counter-clockwise
    confidence: float            # range [0, 1]
    contributing_channels: List[str]   # e.g. ["L","C"]
    timestamp: float

@dataclass
class RadarContact:
    """Display model consumed by UI."""
    angle_deg: float             # range [-180, 180]
    intensity: float             # range [0, 1], decays over time
    born_at: float               # monotonic seconds
```

### 不变量(实现方必须保证)
- `AudioFrame.samples.ndim == 2`
- `AudioFrame.samples.shape[1] == len(AudioFrame.channel_names)`
- `np.all(np.isfinite(AudioFrame.samples))`
- `DirectionEstimate.confidence in [0, 1]`
- `RadarContact.intensity in [0, 1]`

---

## 1. Audio Layer

### 1.1 `src/audio/capture.py`

**职责**:从 WASAPI loopback 设备读取多声道 PCM。

```python
class AudioCapture:
    def __init__(
        self,
        device: Optional[str],       # device name substring or None for default
        sample_rate: int,            # e.g. 48000, > 0
        frame_size: int,             # e.g. 1024, > 0, power of 2 recommended
        channel_layout: str,         # "7.1" | "5.1" | "stereo"
        on_frame: Callable[[AudioFrame], None],  # real-time callback
    ) -> None: ...

    def start(self) -> None: ...     # raises AudioDeviceError on failure
    def stop(self) -> None: ...
    def list_devices(self) -> List[DeviceInfo]: ...
```

**输入范围/约束**:
- `device`: 可空。空表示系统默认渲染设备。
- `sample_rate`: 必须 > 0,典型 48000。
- `frame_size`: 必须 > 0,推荐 2 的幂。
- `channel_layout`: 枚举值。

**输出范围/契约**:
- `on_frame` 在 PortAudio 实时线程被调用,**绝不能阻塞**。
- 传入的 `AudioFrame.samples` 满足上述不变量。

**异常**:
- `AudioDeviceError`:设备不存在、声道数不匹配、采样率不支持。

---

### 1.2 `src/audio/framing.py`

**职责**:维护环形缓冲,提供加窗、重叠帧工具(若需要频谱分析的多帧上下文)。

```python
class RingBuffer:
    def __init__(self, n_channels: int, capacity_samples: int) -> None: ...
    def push(self, samples: np.ndarray) -> None: ...     # shape=(N, n_channels)
    def read_window(self, size: int) -> np.ndarray: ...  # shape=(size, n_channels), oldest first

def hann_window(size: int) -> np.ndarray: ...            # shape=(size,), range [0,1]
```

**输入/输出范围**:
- `RingBuffer.read_window(size)` 返回最近 `size` 个样本(含重叠),shape 固定。
- 缓冲未填满时返回**零填充**(非 None)。

---

## 2. Analysis Layer

### 2.1 `src/analysis/features.py`

**职责**:从 `AudioFrame` 提取特征。**纯函数,无状态**。

```python
def compute_channel_energy(samples: np.ndarray) -> np.ndarray:
    """
    Input:  samples   shape=(frame_size, n_channels), float32, [-1,1]
    Output: energy    shape=(n_channels,), float64, [0, +inf)
    Definition: mean(samples^2) per channel  (== RMS^2)
    """

def compute_band_energy(
    samples: np.ndarray,       # shape=(frame_size, n_channels)
    sample_rate: int,
    bands: List[Band],         # e.g. [(0,200),(200,2000),(2000,8000)]
) -> np.ndarray:
    """
    Output: shape=(n_bands,), float64, [0, +inf)
    Definition: sum of FFT magnitude^2 within each band, summed across channels.
    """
```

**约束**:
- 输入必须 `np.isfinite`。
- 输出永远非负。

---

### 2.2 `src/analysis/noise_floor.py`

**职责**:在线估计本底噪声。

```python
class NoiseFloorEstimator:
    def __init__(
        self,
        n_channels: int,
        attack_ms: int = 50,         # how fast noise floor rises, > 0
        release_ms: int = 5000,      # how slowly it falls, > attack
        sample_rate: int = 48000,
    ) -> None: ...

    def update(self, channel_energy: np.ndarray) -> np.ndarray:
        """
        Input:  shape=(n_channels,), float64, >= 0
        Output: noise_floor shape=(n_channels,), float64, >= 0
                (per-channel estimate, always <= recent input max)
        """
```

**算法**:每声道维护一个值;输入 < 当前值时缓慢下降(release),输入 > 当前值时快速上升(attack)。详见 `05_algorithm.md`。

---

### 2.3 `src/analysis/onset.py`

**职责**:基于 spectral flux 的通用瞬态事件检测。**不绑定任何游戏语义**。

```python
@dataclass
class OnsetEvent:
    timestamp: float
    strength: float        # spectral flux value, >= 0
    band_index: int        # which band the onset is dominant in, 0..n_bands-1

class OnsetDetector:
    def __init__(
        self,
        n_bands: int,
        flux_multiplier: float = 3.0,   # > 0; multiplier over median flux
        refractory_ms: int = 120,       # > 0
        sample_rate: int = 48000,
    ) -> None: ...

    def process(self, frame: AudioFrame, band_energy: np.ndarray) -> Optional[OnsetEvent]:
        """
        Input:  AudioFrame (shape=(frame_size, n_channels))
                band_energy (shape=(n_bands,), >= 0)
        Output: OnsetEvent if flux > flux_multiplier * rolling_median, else None.
                Respects refractory period.
        """
```

**关键性质**:阈值 = `flux_multiplier × 在线中位数`,**完全自适应**,无绝对能量假设。

---

### 2.4 `src/analysis/direction.py` ⭐ 核心

**职责**:VBAP 风格的多声道方位估计。**纯函数,无状态,可单元测试**。

```python
@dataclass
class ChannelLayout:
    names: List[str]                    # e.g. ["L","R","C",...]
    angles: dict                        # {"L": -30, "C": 0, ..., "LFE": None}

def estimate_direction(
    channel_energy: np.ndarray,         # shape=(n_channels,), >= 0
    layout: ChannelLayout,
    noise_floor: np.ndarray,            # shape=(n_channels,), >= 0
    snr_threshold_db: float = 12.0,     # > 0
    ignore_channels: Optional[List[str]] = None,  # e.g. ["C", "LFE"]
    timestamp: float = 0.0,
) -> Optional[DirectionEstimate]:
    """
    Algorithm:
      1. Skip channels in ignore_channels and channels with angle == None (LFE).
      2. Compute SNR per channel = 10*log10(energy / noise_floor).
      3. If max SNR < snr_threshold_db, return None.
      4. Take the two adjacent channels with highest SNR (in angle order).
      5. Weighted-average their angles by SNR-linear weight.
      6. confidence = min(1, (max_snr - threshold) / headroom_db).
    """
```

**输入约束**:
- `channel_energy.shape == noise_floor.shape == (len(layout.names),)`
- 所有元素 >= 0。

**输出约束**:
- 若返回非 None,`angle_deg ∈ [-180, 180]`,`confidence ∈ [0, 1]`。

**测试用例**(详见 `tests/test_direction.py`):
- 全零输入 → 返回 None
- 仅 L 声道有信号 → angle ≈ -30°, contributing_channels = ["L"]
- L + C 等量 → angle ≈ -15°
- 所有声道都低于 SNR 阈值 → None

---

## 3. Tracking Layer

### 3.1 `src/tracking/smoother.py`

**职责**:把瞬时的 `DirectionEstimate` 转成可视化用的持久 `RadarContact`,带衰减。

```python
class ContactSmoother:
    def __init__(
        self,
        decay_ms: int = 400,             # > 0
        angle_smoothing: float = 0.35,   # [0,1]
    ) -> None: ...

    def update(self, estimate: Optional[DirectionEstimate]) -> List[RadarContact]:
        """
        Input:  current DirectionEstimate or None
        Output: list of currently-active RadarContact (intensity > 0)
                After decay_ms with no new event, contact fades out.
        """
```

**关键设计**:
- 角度平滑用**圆周均值**(防止 ±180° 跨界抖动)。
- 同方向(±15° 内)的新事件**刷新**已有 contact,而非新建。
- 不同方向的新事件**新建** contact。

---

## 4. UI Layer

### 4.1 `src/ui/radar_widget.py`

```python
class RadarWidget(QWidget):
    def set_contacts(self, contacts: List[RadarContact]) -> None:
        """Called by UI thread ~60Hz. Replaces current display state."""

    def paintEvent(self, event) -> None:
        """Draws: grid, sweep, zones (front/side/back), contacts."""
```

**输入约束**:`contacts` 中 `intensity ∈ [0,1]`,绘制时映射到 alpha/size。

### 4.2 `src/ui/main_window.py`

```python
class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None: ...

    # Signals (Qt)
    sensitivity_changed: Signal   # emitted when slider moves, float in [0,1]
    layout_changed: Signal        # emitted when user picks 7.1/5.1/stereo
```

**职责**:
- 启动/停止音频捕获
- 把 `contact_queue` 的内容每 16ms 推给 `RadarWidget`
- 提供灵敏度滑杆、声道布局下拉
- 异常弹窗

---

## 5. 配置层

### 5.1 `src/config_loader.py`

```python
@dataclass
class AppConfig:
    audio: AudioConfig
    channel_angles: dict
    direction: DirectionConfig
    onset: OnsetConfig
    tracking: TrackingConfig
    ui: UIConfig

def load_config(
    path: str = "config.yaml",
    profile_path: Optional[str] = None,
) -> AppConfig:
    """
    Validates types/ranges. Raises ConfigError with human-readable message
    on any invalid value.
    """
```

**校验规则示例**:
- `frame_size` 必须 > 0
- `channel_layout` 必须在 `channel_angles` 字典的键里
- `snr_threshold_db` 必须 > 0
- 颜色必须是 `#RRGGBB`

---

## 6. 数据传递约束(跨层)

| 传递方向 | 传递的数据 | 线程 | 同步方式 |
|---|---|---|---|
| capture → processing | `AudioFrame` | audio → proc | `queue.Queue(maxsize=4)`,满则丢旧 |
| processing → ui | `List[RadarContact]` | proc → ui | `queue.Queue(maxsize=2)`,满则丢旧 |
| ui → processing | 灵敏度等参数 | ui → proc | `threading.Event` + 共享只读快照 |

**铁律**:跨线程传递的对象在被消费方使用期间,生产方**不可修改**。所有 dataclass 视为不可变。

---

## 7. 测试要求

- `analysis/` 全部为纯函数,**100% 单元测试覆盖**核心路径
- `direction.py` 的契约测试至少覆盖 6 种方位 + 边界(全零、全低 SNR、单声道、双相邻声道、双跨 180° 声道、全满)
- `capture.py`/`ui/` 不强制单测,但需手动集成测试