# 03 — 模块/方法 输入输出契约

本文档逐模块定义**输入数据类型、范围、单位**与**输出数据类型、范围、含义**。所有实现必须严格遵守此契约。Python 3.9 兼容(类型注解用 `typing.Optional`/`typing.List`,不使用 `X | Y` 运行时语法)。

> **更新历史**
> - 2026-06-21: OnsetDetector 增加了 `n_bands` 参数与 per-band refractory(与原"同频段内抑制"的文档描述对齐);新增线程安全 setter `set_flux_multiplier`;`AudioCapture` 新增 `preflight_device()` 用于启动时声道数自检。

## 0. 共享数据结构(`src/models.py`)

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
class OnsetEvent:
    """A detected transient event from spectral-flux analysis."""
    timestamp: float
    strength: float        # spectral flux value, >= 0
    band_index: int        # which band the onset is dominant in, 0..n_bands-1

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

**职责**:从 WASAPI loopback 设备读取多声道 PCM。启动时做声道数自检。

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

    def preflight_device(self) -> DeviceInfo:
        """Resolve and return the device WITHOUT opening a stream.
        Raises AudioDeviceError if device <expected_channels output channels."""

    def start(self) -> None: ...     # raises AudioDeviceError on failure
    def stop(self) -> None: ...

    @property
    def channel_names(self) -> List[str]: ...
    @property
    def actual_channels(self) -> Optional[int]: ...

    @staticmethod
    def list_devices(host_api: str = "Windows WASAPI") -> List[DeviceInfo]: ...
```

**输入范围/约束**:
- `device`: 可空。空表示系统默认渲染设备。
- `sample_rate`: 必须 > 0,典型 48000。
- `frame_size`: 必须 > 0,推荐 2 的幂。
- `channel_layout`: 枚举值。

**输出范围/契约**:
- `on_frame` 在 PortAudio 实时线程被调用,**绝不能阻塞**。
- 传入的 `AudioFrame.samples` 满足上述不变量。
- `start()` 在打开流之前检查 `device.max_output_channels >= expected_channels`,不满足时抛出带修复指引的 `AudioDeviceError`。

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
    bands: List[Band] = DEFAULT_BANDS,  # 4 bands: 0-200, 200-2k, 2k-8k, 8k-24k
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
        frame_size: int = 1024,
    ) -> None: ...

    def update(self, channel_energy: np.ndarray) -> np.ndarray:
        """
        Input:  shape=(n_channels,), float64, >= 0
        Output: noise_floor shape=(n_channels,), float64, >= 1e-10
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
        n_bands: int = 4,               # must match features.DEFAULT_BANDS length
        flux_multiplier: float = 3.0,   # > 0; multiplier over median flux
        refractory_ms: int = 120,       # > 0; PER-BAND refractory
        median_window_frames: int = 60, # ~1 second at 60 Hz frame rate
    ) -> None: ...

    def set_flux_multiplier(self, value: float) -> None:
        """Thread-safe setter (UI thread calls this)."""

    def get_flux_multiplier(self) -> float:
        """Thread-safe getter (UI readback / tests)."""

    def process(self, frame: AudioFrame, band_energy: np.ndarray) -> Optional[OnsetEvent]:
        """
        Input:  AudioFrame (shape=(frame_size, n_channels))
                band_energy (shape=(n_bands,), >= 0)
        Output: OnsetEvent if flux > flux_multiplier * rolling_median, else None.
                Refractory is PER-BAND: an onset on band 2 does NOT suppress
                a near-simultaneous onset on band 0.
        """
```

**关键性质**:
- 阈值 = `flux_multiplier × 在线中位数`,**完全自适应**,无绝对能量假设。
- `flux_multiplier` 受锁保护,可由 UI 线程通过 `set_flux_multiplier()` 安全修改,**禁止**直接写 `._flux_multiplier`。
- refractory 按 band 独立(符合 `05_algorithm.md` 3.3 节"同频段内"的描述)。

---

### 2.4 `src/analysis/direction.py` ⭐ 核心

**职责**:VBAP 风格的多声道方位估计。**纯函数,无状态,可单元测试**。

```python
def estimate_direction(
    channel_energy: np.ndarray,         # shape=(n_channels,), >= 0
    layout: ChannelLayout,
    noise_floor: np.ndarray,            # shape=(n_channels,), >= 0
    snr_threshold_db: float = 12.0,     # > 0
    ignore_channels: Optional[List[str]] = None,  # None -> default ["C","LFE"]; [] -> ignore nothing
    headroom_db: float = 18.0,
    timestamp: float = 0.0,
) -> Optional[DirectionEstimate]:
    """
    Algorithm:
      1. Skip channels with angle == None (LFE) and those in ignore_channels.
      2. Compute SNR per channel = 10*log10(energy / noise_floor).
      3. If no channel clears snr_threshold_db, return None.
      4. Pick the above-threshold channel with the highest SNR (A).
      5. Among the remaining above-threshold channels, pick the angularly
         nearest one to A (B).
      6. Weighted-interpolate A and B's angles by SNR-linear weights.
      7. confidence = (A_snr - threshold) / headroom_db, clamped to [0,1].
    """
```

**输入约束**:
- `channel_energy.shape == noise_floor.shape == (len(layout.names),)`
- 所有元素 >= 0。

**输出约束**:
- 若返回非 None,`angle_deg ∈ [-180, 180]`,`confidence ∈ [0, 1]`。

**默认行为**:`ignore_channels=None` 时默认忽略 `["C","LFE"]`(见 `05_algorithm.md` 4.3)。传 `[]` 可关闭所有忽略。

**测试用例**(详见 `tests/test_direction.py`):
- 全零输入 → 返回 None
- 仅 L 声道有信号 → angle ≈ -30°
- L + R 等量 → angle ≈ 0°
- Lb + Rb 等量(跨 ±180)→ angle ≈ ±180°
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
        clock: Optional[Callable[[], float]] = None,  # injectable for tests
    ) -> None: ...

    def update(self, estimate: Optional[DirectionEstimate]) -> List[RadarContact]:
        """
        Input:  current DirectionEstimate or None
        Output: list of currently-active RadarContact (intensity > 0.02)
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

    def advance_sweep(self, dt_seconds: float) -> None:
        """Advance the radar sweep animation. Call from a timer."""

    def paintEvent(self, event) -> None:
        """Draws: grid, sweep, zones (front/side/back), contacts."""
```

**输入约束**:`contacts` 中 `intensity ∈ [0,1]`,绘制时映射到 alpha/size。

### 4.2 `src/ui/main_window.py`

```python
class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None: ...

    # Internal signal: slider value -> re-maps snr_threshold_db and flux_multiplier.
    # HIGHER slider value = MORE sensitive = LOWER thresholds.
```

**职责**:
- 启动/停止音频捕获
- 把 `contact_queue` 的内容每 16ms 推给 `RadarWidget`
- 提供灵敏度滑杆(语义:拖右更敏感)
- 异常记日志,不静默吞

**并发契约**(重要):
- UI 线程**绝不直接**修改处理线程的状态。
- 可调参数(`snr_threshold_db`)由 UI 写入**加锁快照**,处理线程读取。
- `flux_multiplier` 通过 `OnsetDetector.set_flux_multiplier()` 修改(内部有锁),**禁止**直接访问私有属性。

---

## 5. 配置层

### 5.1 `src/config_loader.py`

```python
@dataclass
class AppConfig:
    audio: AudioConfig
    channel_angles: dict
    channel_layout_obj: ChannelLayout
    direction: DirectionConfig
    onset: OnsetConfig
    tracking: TrackingConfig
    ui: UIConfig

def load_config(
    path: Optional[str] = "config.yaml",
    profile_path: Optional[str] = None,
) -> AppConfig:
    """
    Priority: profile > config.yaml (if exists) > _HARD_DEFAULTS.
    _HARD_DEFAULTS is COMPLETE: every field (including channel_angles and
    UI colors) is present, so the app runs even with path=None and no
    profile. Raises ConfigError on any invalid value.
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
| ui → processing (snr) | `float` | ui → proc | `threading.Lock` 保护的 `_params` 快照 |
| ui → onset (flux) | `float` | ui → proc | `OnsetDetector` 内部的锁 + `set_flux_multiplier()` |

**铁律**:跨线程传递的对象在被消费方使用期间,生产方**不可修改**。所有 dataclass 视为不可变。UI 线程**绝不**直接写处理线程对象的私有属性。

---

## 7. 错误处理

| 场景 | 处理 |
|---|---|
| WASAPI 设备不存在 | `preflight_device()` / `start()` 抛 `AudioDeviceError`,main 打印可用设备列表后退出 |
| 设备声道数 < 配置 | `AudioDeviceError` 含明确修复指引(配 Windows 7.1) |
| 实际声道数 ≠ 配置(运行中) | 启动时 `start()` 自检已拦截;回调里仍吞异常保活 |
| 处理线程异常 | `logger.exception()` 记录,线程不退出 |
| UI 线程异常 | 状态栏显示 + logger 记录 |

---

## 8. 测试要求

- `analysis/` 全部为纯函数,**100% 单元测试覆盖**核心路径
- `direction.py` 的契约测试覆盖 6 种方位 + 边界(全零、全低 SNR、单声道、双相邻声道、双跨 180° 声道、全满)
- `onset.py` 覆盖:首帧无事件、per-band refractory、`set_flux_multiplier` 线程安全与拒绝非法值
- `config_loader.py` 覆盖:无 config 时硬编码兜底、profile 合并、非法布局报错
- `smoother.py` 用注入时钟覆盖衰减/合并/平滑
- `capture.py`/`ui/` 不强制单测,但 `tools/smoke_test.py` 做端到端冒烟