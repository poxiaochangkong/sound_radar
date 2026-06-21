# 05 — 核心算法说明

本文档详细说明声音雷达使用的所有算法,确保每个公式都可追溯到 `03_module_io.md` 中的契约。

## 1. 多声道能量特征(`features.py`)

### 1.1 每声道 RMS²

$$
E_c = \frac{1}{N} \sum_{n=0}^{N-1} x_c^2[n]
$$

其中 $c$ 为声道索引,$N$ 为帧长。等价于 numpy 的 `(samples**2).mean(axis=0)`。

**输出**:`shape=(n_channels,)`,`dtype=float64`,`range=[0, +inf)`。

### 1.2 子带能量(用于 onset 检测)
对每帧做 FFT,把幅度谱平方后按频带累加:

$$
E_{band,k} = \sum_{f \in band_k} |X(f)|^2
$$

默认频带(`DEFAULT_BANDS`,4 个):
- Band 0: 0–200 Hz(低频)
- Band 1: 200–2000 Hz(中频,脚步主能量)
- Band 2: 2000–8000 Hz(高频,枪声能量)
- Band 3: 8000–Nyquist(空气/瞬时)

实现:`np.fft.rfft`,跨声道求和。`OnsetDetector` 的 `n_bands` 参数必须等于 `DEFAULT_BANDS` 的长度。

---

## 2. 自适应本底噪声(`noise_floor.py`)

### 2.1 算法:每声道 attack/release 包络

对每声道 $c$ 维护一个标量 $N_c$,每帧更新:

**Release(输入低 → 噪声估计缓慢下降)**
$$
N_c \leftarrow N_c - \alpha_{rel} \cdot (N_c - E_c) \quad \text{if } E_c < N_c
$$

**Attack(输入高 → 噪声估计快速上升)**
$$
N_c \leftarrow N_c + \alpha_{atk} \cdot (E_c - N_c) \quad \text{if } E_c \geq N_c
$$

**系数**($T$ 为帧时长秒):
$$
\alpha = 1 - \exp\left(-\frac{T}{\tau}\right), \quad \tau_{atk}=0.05\text{s}, \tau_{rel}=5\text{s}
$$

### 2.2 不变量
- $N_c \geq 0$
- 平稳期 $N_c$ 跟踪本底噪声平均值
- 突发事件时 $N_c$ 缓慢爬升,不会瞬间吞掉事件

### 2.3 数值下限
为防止 log(0),实现时保证 $N_c \geq 10^{-10}$。

### 2.4 已知限制
强瞬态声音可能快速抬高 $N_c$(attack 50ms),导致紧随其后的连续脚步 SNR 降低。**未来增强**:onset 触发帧暂停更新 noise floor,或改用 rolling percentile。当前实现是 MVP 的有意识简化选择。

---

## 3. Spectral Flux Onset 检测(`onset.py`)

### 3.1 Spectral Flux 定义
对每帧频谱 $|X_k(f)|$,与上一帧频谱求**仅正差值**之和:

$$
\text{Flux}_k = \sum_f \max\left(0, |X_k(f)| - |X_{k-1}(f)|\right)
$$

仅取正差值,只对"能量增加"敏感,这是 onset 的本质。

### 3.2 自适应阈值
维护 Flux 的**滚动中位数** $\text{Med}_k$(窗口 ~1 秒)。当:

$$
\text{Flux}_k > \text{flux\_multiplier} \times \text{Med}_k
$$

触发 onset 事件。

`flux_multiplier` 受锁保护,UI 线程通过 `set_flux_multiplier()` 修改。

### 3.3 Per-Band 不应期(refractory)
**每个频段**独立维护 `_last_onset_per_band[band_index]`。同一频段在 `refractory_ms` 内不再触发,**不同频段互不影响**。

这样设计的好处:枪声(高频,bias to band 2)和脚步(低频,bias to band 0/1)即使同时发生,也不会因为一个触发而把另一个屏蔽掉。

### 3.4 为什么不用绝对阈值?
不同游戏、不同 map、不同时间段的"安静段能量"差异巨大。绝对阈值(`flux > 0.5`)在某些游戏永远触发,在某些游戏从不触发。**中位数倍数**对所有情况自适应。

---

## 4. VBAP 方位估计(`direction.py`)⭐

### 4.1 VBAP 原理
**Vector Base Amplitude Panning**(向量基振幅平移):声源方位 = 两个相邻扬声器方位的能量加权向量。

### 4.2 算法步骤

**Step 1:选择候选声道**
- 排除 `ignore_channels`(默认 `["C","LFE"]`)
- 排除 `angle is None` 的声道(`LFE`)

**Step 2:计算每声道 SNR**
$$
\text{SNR}_c = 10 \log_{10}\left(\frac{E_c + \epsilon}{N_c + \epsilon}\right) \quad [\text{dB}]
$$
$\epsilon = 10^{-10}$ 防止除零。

**Step 3:SNR 阈值门限**
- 没有任何候选声道超过 `snr_threshold_db` → 返回 `None`

**Step 4:在"通过阈值的声道"中找 A 和 B**
- A = SNR 最高的通过阈值的声道
- B = 在剩余通过阈值的声道中,**角度上离 A 最近**的那个
- 注意:B 只在通过阈值的声道中找(否则会把无信号的最近邻居拉进来,导致角度偏向单一声道)

**Step 5:加权插值角度**
把 SNR 从 dB 转线性作为权重:
$$
w_c = \max(0, \text{SNR}_c - \text{snr\_threshold\_db})
$$
$$
\theta = \frac{w_A \theta_A + w_B \theta_B}{w_A + w_B}
$$

**关键:圆周角处理**
- 如果 $|\theta_A - \theta_B| > 180$,用最短符号差(在 `[-180,180]` 内)插值,最后归一到 `[-180,180]`
- 例:$A = Lb(-150°)$, $B = Rb(+150°)$ → 真实间隔是 60°(跨过 ±180),中点 ±180°

**Step 6:置信度**
$$
\text{confidence} = \min\left(1, \frac{\max_c \text{SNR}_c - \text{snr\_threshold\_db}}{\text{headroom\_db}}\right)
$$
`headroom_db` 默认 18dB:超过阈值 18dB 就满置信度。

### 4.3 为什么默认忽略 C 声道?
许多 FPS 把**非定位声**(UI、菜单音乐、环境循环)混到 C 声道。这些声音会被误判为"正前"。所以默认 ignore C,只信任 L/R/Ls/Rs/Lb/Rb。

**代价**:正前方分辨率下降到 L/R 之间(±15°),因为正前事件被强行归到 L 或 R。

如果某游戏确实把脚步也混到 C,可以设 `ignore_center_channel: false`,正前分辨率恢复到 ±5°。

### 4.4 默认行为
`estimate_direction(ignore_channels=None)` → 忽略 `["C","LFE"]`。
`estimate_direction(ignore_channels=[])` → 关闭所有忽略(此时 C 会被用于方位估计)。

---

## 5. 时序平滑与衰减(`smoother.py`)

### 5.1 Contact 生命周期
- `born_at`:事件首次触发时间
- `intensity`:从 1.0 开始,按指数衰减
- 衰减公式(半衰期 $T_{1/2} = \text{decay\_ms}$):
$$
I(t) = I_0 \cdot 2^{-\frac{t - \text{born\_at}}{T_{1/2}}}
$$
- $I(t) < 0.02$ 时移除 contact。

### 5.2 同方向刷新 vs 新建
- 新事件 angle 与已有 contact 的 angle 差 $< 15°$ → 刷新该 contact(重置 `born_at`,低通滤波 angle)
- 否则新建 contact

### 5.3 角度低通平滑
$$
\theta_{smooth} \leftarrow (1-\alpha) \theta_{new} + \alpha \theta_{prev}
$$
但用**圆周差**(防 ±180 跳变),$\alpha = \text{angle\_smoothing}$。

---

## 6. 线程与队列

### 6.1 帧队列(capture → processing)
- `queue.Queue(maxsize=4)`
- `put(block=False)`,满则 `except queue.Full` → 丢弃**最旧帧**后重试
- 理由:实时雷达宁可漏一帧,不能阻塞音频回调

### 6.2 Contact 队列(processing → ui)
- `queue.Queue(maxsize=2)`
- 同样丢旧策略

### 6.3 参数同步(ui → processing)
| 参数 | 同步方式 |
|---|---|
| `snr_threshold_db` | UI 线程写入 `MainWindow._params`(由 `threading.Lock` 保护),处理线程读取快照 |
| `flux_multiplier` | UI 线程调用 `OnsetDetector.set_flux_multiplier()`(对象内部锁),**绝不**直接写私有属性 |
| 其他(`ignore_channels` 等) | 启动时由 config 一次性决定,运行期不可变 |

---

## 7. 灵敏度滑杆(单旋钮调参)

UI 滑杆 0–100,**数值越大越敏感**(即越容易触发)。

### 7.1 映射公式
```
sensitivity_factor = 0.5 + 1.5 * (slider_value / 100)   # 0.5 at 0, 2.0 at 100
effective_snr_threshold  = base_snr  / sensitivity_factor
effective_flux_multiplier = base_flux / sensitivity_factor
```

### 7.2 直觉
| 滑杆值 | 含义 | 行为 |
|---|---|---|
| 100(最右) | 最敏感 | 阈值最低,远距离/微弱声也能触发;可能误报 |
| 50(中) | 标准 | 适合多数游戏 |
| 0(最左) | 最不敏感 | 阈值最高,只触发明显的近距离声;可能漏报 |

### 7.3 调参建议
1. 进入游戏空旷处,无人走动 → 调到刚好**无任何雷达点**(通常 30–50)
2. 让朋友在 10 米外走动 → 调到能**稳定显示**其方位(通常 50–70)
3. 长时间游戏后微调

---

## 8. 参数总览(对应 config.yaml)

| 参数 | 默认值 | 含义 | 调节影响 |
|---|---|---|---|
| `audio.frame_size` | 1024 | 帧长 | 大 → 延迟高但能量估计稳 |
| `audio.sample_rate` | 48000 | 采样率 | 匹配系统 |
| `direction.snr_threshold_db` | 12.0 | 事件触发 SNR 门限(滑杆=50 时的基准) | 高 → 不敏感;低 → 易误报 |
| `direction.ignore_center_channel` | true | 忽略 C 声道 | 关掉 → 某些游戏前向更准 |
| `onset.flux_multiplier` | 3.0 | onset 阈值倍数(滑杆=50 时的基准) | 高 → 少误报;低 → 多触发 |
| `onset.refractory_ms` | 120 | **per-band** 最短间隔 | 高 → 同频段合并快速连击 |
| `tracking.decay_ms` | 400 | contact 衰减半衰期 | 高 → 雷达点久留 |
| `tracking.angle_smoothing` | 0.35 | 角度低通系数 | 高 → 平滑但响应慢 |

---

## 9. 已知限制(文档透明)

1. **无法识别上下方位**:无高度声道。这是多声道方案的物理限制,无算法能解。
2. **C 声道非定位声干扰**:默认忽略 C,代价是正前方分辨率下降到 L/R 之间 ±15°(因为正前事件被强行归到 L 或 R)。可通过 `ignore_center_channel: false` 反转。
3. **混响/回声会模糊方位**:游戏内大空间混响会让能量分布扩散,VBAP 输出可能漂移。
4. **同时多个事件**:若两声源同时在不同方位,VBAP 输出会偏向能量大者,这是单帧单源假设的局限。未来可扩展为多事件分离。
5. **noise floor 可能被强瞬态抬高**:attack 50ms 较快,连续脚步/快速枪声可能漏检。未来增强:onset 触发帧暂停更新,或改用 rolling percentile。

---

## 10. 相关文档
- `03_module_io.md` — 算法对应的 I/O 契约
- `06_calibration.md` — 如何为特定环境/游戏调参