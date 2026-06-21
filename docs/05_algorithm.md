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

默认频带:
- Band 0: 0–200 Hz(低频)
- Band 1: 200–2000 Hz(中频,脚步主能量)
- Band 2: 2000–8000 Hz(高频,枪声能量)
- Band 3: 8000–Nyquist(空气/瞬时)

实现:`scipy.signal.spectrogram` 或直接 `np.fft.rfft`,跨声道求和。

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

---

## 3. Spectral Flux Onset 检测(`onset.py`)

### 3.1 Spectral Flux 定义
对每帧频谱 $|X_k(f)|^2$,与上一帧频谱求**仅正差值**之和:

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

### 3.3 不应期(refractory)
同频段内上次事件后 `refractory_ms` 内不再触发,防止同一脚步被多次检测。

### 3.4 为什么不用绝对阈值?
不同游戏、不同 map、不同时间段的"安静段能量"差异巨大。绝对阈值(`flux > 0.5`)在某些游戏永远触发,在某些游戏从不触发。**中位数倍数**对所有情况自适应。

---

## 4. VBAP 方位估计(`direction.py`)⭐

### 4.1 VBAP 原理
**Vector Base Amplitude Panning**(向量基振幅平移):声源方位 = 两个相邻扬声器方位的能量加权向量。

### 4.2 算法步骤

**Step 1:选择可用声道**
- 排除 `ignore_channels`(默认 `["C","LFE"]`)
- 排除 `angle is None` 的声道(`LFE`)

**Step 2:计算每声道 SNR**
$$
\text{SNR}_c = 10 \log_{10}\left(\frac{E_c + \epsilon}{N_c + \epsilon}\right) \quad [\text{dB}]
$$
$\epsilon = 10^{-10}$ 防止除零。

**Step 3:SNR 阈值门限**
- $\max_c \text{SNR}_c < \text{snr\_threshold\_db}$ → 返回 `None`(没有可信事件)

**Step 4:找最大两个相邻声道**
- 按 angle 排序所有候选声道
- 选 SNR 最高的声道 $A$
- 在角度环上找 $A$ 的两个邻居中 SNR 较高者 $B$

**Step 5:加权插值角度**
把 SNR 从 dB 转线性作为权重:
$$
w_c = \max(0, \text{SNR}_c - \text{snr\_threshold\_db})
$$
$$
\theta = \frac{w_A \theta_A + w_B \theta_B}{w_A + w_B}
$$

**关键:圆周角处理**
- 如果 $|\theta_A - \theta_B| > 180$,把负角 +360 后再插值,最后归一到 $[-180,180]$
- 例:$A = Lb(-150°)$, $B = Rb(+150°)$ → 真实间隔是 60°(跨过 ±180),不是 300°
- 正确做法:把 $-150$ 加 360 变成 $210$,然后与 $150$ 插值 → 中点 $180/-180$

**Step 6:置信度**
$$
\text{confidence} = \min\left(1, \frac{\max_c \text{SNR}_c - \text{snr\_threshold\_db}}{\text{headroom\_db}}\right)
$$
`headroom_db` 默认 18dB:超过阈值 18dB 就满置信度。

### 4.3 为什么默认忽略 C 声道?
许多 FPS 把**非定位声**(UI、菜单音乐、环境循环)混到 C 声道。这些声音会被误判为"正前"。所以默认 ignore C,只信任 L/R/Ls/Rs/Lb/Rb。

如果某游戏确实把脚步也混到 C,可以关掉 ignore(在 config 里设置 `ignore_center_channel: false`)。

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
- 新事件 angle 与已有 contact 的 angle 差 $< 15°$ → 刷新该 contact(重置 `born_at`,设新 angle)
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
- 灵敏度等参数由 UI 滑杆修改,processing 线程定期读取共享只读快照
- 用 `threading.Lock` 保护快照指针,锁内只做引用拷贝,纳秒级

---

## 7. 参数总览(对应 config.yaml)

| 参数 | 默认值 | 含义 | 调节影响 |
|---|---|---|---|
| `audio.frame_size` | 1024 | 帧长 | 大 → 延迟高但能量估计稳 |
| `audio.sample_rate` | 48000 | 采样率 | 匹配系统 |
| `direction.snr_threshold_db` | 12.0 | 事件触发 SNR 门限 | 高 → 不敏感;低 → 易误报 |
| `direction.ignore_center_channel` | true | 忽略 C 声道 | 关掉 → 某些游戏前向更准 |
| `onset.flux_multiplier` | 3.0 | onset 阈值倍数 | 高 → 少误报;低 → 多触发 |
| `onset.refractory_ms` | 120 | 同频段最短间隔 | 高 → 合并快速连击 |
| `tracking.decay_ms` | 400 | contact 衰减半衰期 | 高 → 雷达点久留 |
| `tracking.angle_smoothing` | 0.35 | 角度低通系数 | 高 → 平滑但响应慢 |

---

## 8. 已知限制(文档透明)

1. **无法识别上下方位**:无高度声道。这是多声道方案的物理限制,无算法能解。
2. **C 声道非定位声干扰**:默认忽略 C,代价是正前方分辨率下降到 L/R 之间 ±15°(因为正前事件被强行归到 L 或 R)。
3. **混响/回声会模糊方位**:游戏内大空间混响会让能量分布扩散,VBAP 输出可能漂移。
4. **同时多个事件**:若两声源同时在不同方位,VBAP 输出会偏向能量大者,这是单帧单源假设的局限。未来可扩展为多事件分离。

---

## 9. 相关文档
- `03_module_io.md` — 算法对应的 I/O 契约
- `06_calibration.md` — 如何为特定环境/游戏调参