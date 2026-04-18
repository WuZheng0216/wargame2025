# Low-Cost Trajectory Diagnostics

这套诊断链的目的不是先改策略，而是先回答三个问题：

1. 低成本攻击弹最近距离蓝方目标到底有多近。
2. 发射时红方拿到的目标位置和蓝方真值位置偏差有多大。
3. 现象更像“预测器还有优化空间”，还是“蓝方拦截/规避已经主导结果”。

## 1. 开启逐帧轨迹记录

单局诊断前，在 PowerShell 里设置：

```powershell
$env:TRAJECTORY_DIAGNOSTICS_ENABLED="1"
$env:TRAJECTORY_DIAGNOSTICS_INTERVAL_SECONDS="1"
```

如果你同时还想看 `scene` 里低成本弹预测器的调试输出，可以额外开：

```powershell
$env:JSQLSIM_ONESHOT_DEBUG="1"
```

然后正常跑一局仿真。

## 2. 诊断文件会写到哪里

单局模式默认写到：

- `test/diagnostics/trajectory_RED_*.jsonl`
- `test/diagnostics/trajectory_BLUE_*.jsonl`

batch 模式下会跟着 `RUN_OUTPUT_ROOT` 进入每个 `run_xxxx/diagnostics/`。

## 3. 离线分析

跑完后执行：

```powershell
& C:\Users\Tk\.conda\envs\scene\python.exe .\analyze_lowcost_trajectory_diagnostics.py
```

也可以指定文件：

```powershell
& C:\Users\Tk\.conda\envs\scene\python.exe .\analyze_lowcost_trajectory_diagnostics.py --red-file .\test\diagnostics\trajectory_RED_xxx.jsonl --blue-file .\test\diagnostics\trajectory_BLUE_xxx.jsonl
```

输出会默认保存到：

- `analysis_lowcost_trajectory.json`

## 4. 结果怎么看

重点看这些字段：

- `entered_lowcost_terminal_window_count`
  - 进入低成本弹 5km 末制导窗口的次数
- `near_miss_predictor_tunable_count`
  - 接近目标但没有进入 5km 窗口，说明预测器还有继续优化空间
- `median_closest_approach_m`
  - 中位最近距离
- `median_launch_observation_error_m`
  - 发射时红方观测位置和蓝方真值位置的典型偏差
- `classification_counts`
  - 每枚导弹的总体分类

## 5. 推荐解释口径

- 如果很多导弹都在 `5km ~ 15km` 附近掠过：
  - 说明预测器还有明显优化空间
- 如果不少导弹已经进入 `<=5km` 窗口但仍然战果差：
  - 更像蓝方拦截/规避主导
- 如果发射时观测误差本身就很大：
  - 先解决航迹新鲜度和引导/侦察问题，再谈继续调预测器

## 6. 诊断结束后记得关掉

大批量实验前建议关闭诊断，避免多余 I/O：

```powershell
Remove-Item Env:TRAJECTORY_DIAGNOSTICS_ENABLED -ErrorAction SilentlyContinue
Remove-Item Env:TRAJECTORY_DIAGNOSTICS_INTERVAL_SECONDS -ErrorAction SilentlyContinue
Remove-Item Env:JSQLSIM_ONESHOT_DEBUG -ErrorAction SilentlyContinue
```
