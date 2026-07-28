# 调试、标定与测试手册

本文件集中记录运行本项目时最常用的参数解释、传统视觉调参方法、测试程序用法和激光光轴标定流程。所有测试输出都会写入 `test_output/`，该目录不提交到 Git。

## 1. 调试顺序

建议不要一开始就运行完整瞄准链路，而是按以下顺序排查：

1. 使用 `test_hik_camera` 确认相机能正常取图。
2. 使用 `test_laser_model` 确认模型检测框和 ROI 映射正确。
3. 使用 `test_device_center` 或 `test_traditional_center` 调整红光提取参数。
4. 使用 `test_cylinder_pose` 确认红光方块、PnP 和目标距离正确。
5. 使用 `test_coordinate_transformer`、`test_world_target_observation` 和 `test_world_target_tracker` 验证坐标与跟踪逻辑。
6. 完成激光光轴标定后，再运行 `laser_aim`；确认输出稳定后才使用 `--send`。

## 2. 关键配置参数

主配置文件是 `config/laser.json`。长度单位统一为米；相机坐标遵循 OpenCV 约定：`+X` 向图像右侧、`+Y` 向图像下侧、`+Z` 向镜头前方。

### 相机与手眼参数

| 配置项 | 含义 | 注意事项 |
| --- | --- | --- |
| `camera.sn` | 海康相机序列号 | 必须与实际连接设备一致。 |
| `camera.image_width` / `image_height` | 标定和运行使用的图像分辨率 | 分辨率变更后，内参通常不能直接沿用。 |
| `camera.camera_matrix` | 相机内参矩阵 | 必须匹配当前相机、镜头和分辨率。 |
| `camera.distortion_coefficients` | 镜头畸变参数 | PnP 使用该参数修正畸变。 |
| `handeye.R_camera2gimbal` | 相机坐标系到云台坐标系的旋转 | 需结合机械安装关系标定。 |
| `handeye.t_camera2gimbal_m` | 相机原点在云台坐标系中的平移 | 单位为米，方向必须与坐标轴定义一致。 |
| `handeye.R_gimbal2imubody` | 云台到 IMU-body 的旋转 | 与电控反馈的姿态定义保持一致。 |

### 目标几何与激光参数

| 配置项 | 含义 | 注意事项 |
| --- | --- | --- |
| `pnp.light_square_width_m` / `light_square_height_m` | 单个发光方块的真实尺寸 | 尺寸误差会直接影响 PnP 距离与姿态。 |
| `pnp.emitting_face_radius_m` | 发光面中心到圆柱中心轴线的距离 | 不是中间窄圆柱的半径。 |
| `pnp.ring_center_separation_m` | 上、下两圈发光方块中心距离 | 应实测。 |
| `pnp.markers_per_ring` | 每圈方块数量 | 与真实装置结构一致。 |
| `pnp.ring_alignment_deg` | 下圈相对上圈的圆周错位角 | 对齐为 `0`，错开半格时按实际角度填写。 |
| `laser.t_laser_in_camera_m` / `R_laser2camera` | 激光器相对相机的外参 | 只适用于“平行光轴”回退模型。 |
| `laser.beam_line_in_camera` | 标定得到的真实激光光束线 | 光束标定完成后将 `enabled` 设为 `1`。 |

### 检测、瞄准与跟踪参数

| 配置项 | 作用 | 调整建议 |
| --- | --- | --- |
| `detector.engine` | TensorRT 引擎路径 | 使用与当前 CUDA、TensorRT、GPU 匹配的本机引擎。 |
| `detector.confidence_threshold` | YOLO 置信度阈值 | 误检多时调高，漏检多时调低。 |
| `detector.nms_threshold` | 检测框去重阈值 | 越小越积极地合并重叠框。 |
| `detector.min_content_overlap` | ROI 落在有效图像区的最小比例 | 用于过滤 letterbox 填充区域的假框。 |
| `aim.max_reprojection_error_px` | PnP 重投影误差上限 | 过大容易接受错误姿态，过小容易拒绝正常观测。 |
| `aim.static_compensation` | 距离相关的经验角度补偿 | 应先确保 PnP、坐标系和激光外参正确，再使用。 |
| `tracking.*` | 目标初始化、测量噪声、门限和预测参数 | 优先测量真实延迟和抖动，再调整门限与噪声。 |

## 3. 传统红光视觉调参

传统视觉只处理 YOLO 给出的 ROI。推荐顺序是：先调 HSV，再调形态学处理，再调连通域筛选，最后调空间分组。每次只改一类参数，并保存对应输出图像。

### HSV 红色阈值

配置位置：`device_center.hsv`。

```json
"lower_red_1": [0, 45, 140],
"upper_red_1": [30, 255, 255],
"lower_red_2": [165, 80, 120],
"upper_red_2": [179, 255, 255]
```

OpenCV 的 HSV 中，红色跨越色相范围两端，因此必须使用两段阈值并合并结果。

- **H（色相）**：真实红光偏橙或偏暖但被漏检时，适当放宽 H；红色背景和橙色杂光进入时，收紧 H。
- **S（饱和度）**：下限调高会排除白光反射和灰色物体，但远处、过曝或雾中的红灯也可能被漏掉；下限调低则相反。
- **V（亮度）**：下限调高只保留更亮的发光区域；下限调低能保留暗处或远距离红光，但会带来更多暗红背景和噪声。

如果目标同时包含紫色灯，可开启 `enable_purple` 并设置 `lower_purple` / `upper_purple`。

### 形态学闭运算

配置项：`device_center.morphology.close_kernel_size`。

- `1`：不修补；
- `3`：推荐起点，可连接轻微断裂的灯块；
- `5`：修补更强；
- `7` 及以上：容易将相邻的独立灯块粘连。

同一盏灯总被分成多个小块时可调大；相邻灯总被合并时应调小。偶数会被程序调整为下一个奇数，以保证形态学核有明确中心。

### 连通域筛选

配置位置：`device_center.components`。

| 参数 | 含义与调法 |
| --- | --- |
| `min_area` | 最小像素面积。噪点多时调大，远处真实小灯被过滤时调小。 |
| `max_area_ratio` | 单个候选最多占 ROI 的面积比例。红色背景或反光被选中时调小，近距离过曝灯被过滤时调大。 |
| `min_side` | 旋转外接矩形短边下限。调大可过滤细小噪声，调小可保留模糊小灯。 |
| `max_aspect_ratio` | 长短边比例上限。调小可过滤红色线条和反射条纹，调大可接受透视导致的拉长灯块。 |
| `min_fill_ratio` | 实际白色面积与旋转外接矩形面积之比。调高可排除细线和零散噪点，调低可保留破碎灯块。 |

### 空间分组

配置位置：`device_center.grouping`。程序会将距离和结构接近的候选灯块组成一套检测装置。

| 参数 | 含义与调法 |
| --- | --- |
| `link_factor` | 连接距离相对典型灯块尺寸的倍数。调大更容易合并远处候选，调小更容易拆开不同装置。 |
| `min_group_points` | 一个候选组的最少灯块数。调大更严格，调小可容忍缺灯或遮挡。 |
| `min_points_per_layer` | 上、下两层各自的最少灯块数。调小可容忍不完整结构，但中心稳定性会下降。 |
| `min_layer_separation_size_ratio` | 上下两层中心距离相对灯块尺寸的下限。调大可避免将同一层杂点误判为两层。 |
| `min_separation_ratio` | 上下两层分离程度的下限。调大只接受层次更清晰的结构。 |
| `max_horizontal_offset_ratio` | 上、下层中心允许的水平错位。调小更严格，调大可容忍透视和装置旋转。 |

## 4. 常用测试程序

| 目标 | 命令 | 用途 |
| --- | --- | --- |
| 相机取图 | `./build/test/test_hik_camera` | 验证海康相机是否能够枚举、连接和读取图像。 |
| 模型检测 | `./build/test/test_laser_model <image_path> [config_path] [output_directory]` | 输出模型输入坐标、映射回原图的 ROI 和每个裁剪结果。先检查模型输入框，再检查原图 ROI 框。 |
| 红光中心 | `./build/test/test_device_center <roi_image_path> [output_image_path]` | 不经过 TensorRT，直接验证单张 ROI 的红光中心提取。 |
| 传统视觉可视化 | `./build/test/test_traditional_center [config_path] [image_path] [output_directory] [--camera]` | 输出各处理阶段图像；加入 `--camera` 可实时读取海康相机，按 `Esc` 或 `q` 退出。 |
| 圆柱 PnP | `./build/test/test_cylinder_pose <roi_image> [config_path] [output_image] [roi_x roi_y]` | 验证方块角点、PnP 重投影和相机坐标系中的三维目标中心。若输入来自原图裁剪，传入裁剪左上角坐标。 |
| 坐标变换 | `./build/test/test_coordinate_transformer` | 不接相机硬件，验证相机、云台与世界坐标变换。 |
| 跟踪与观测 | `./build/test/test_world_target_observation`、`./build/test/test_world_target_tracker` | 验证世界坐标观测质量和目标跟踪逻辑。 |
| 激光绝对瞄准 | `./build/test/test_absolute_laser_aim` | 验证激光外参与绝对 yaw / pitch 解算。 |
| 激光光轴标定 | `./build/test/calibrate_laser_boresight [config_path] [output_json_path]` | 采集棋盘格中心并拟合真实光束线。 |

## 5. 激光光轴（boresight）标定

本流程主要参考华北理工大学 HORIZON 战队的 boresight 同轴标定思路。它拟合的是**激光器在相机坐标系中的真实光束线**，用于处理激光光轴不平行和近距离视差；它不替代相机内参、手眼外参或目标 PnP 尺寸标定。

### 标定前准备

1. 先确认 `camera.camera_matrix`、`camera.distortion_coefficients`、`laser.t_laser_in_camera_m` 和 `laser.R_laser2camera` 已填写且合理。
2. 制作平整棋盘格板；`chessboard_inner_corners` 是内角点数，不是方格数量。
3. 用尺子实测一个方格边长，填写 `laser.boresight_calibration.square_size_m`。
4. 确保棋盘格完整入画、角点清晰，避免反光、弯曲和明显遮挡。
5. 标定过程中不要移动相机和激光器之间的刚性安装关系。

### 运行与采样

```bash
./build/test/calibrate_laser_boresight config/laser.json
```

在约 `1.0 m`、`1.6 m`、`2.4 m`、`3.2 m`、`4.0 m`、`5.0 m` 等多个距离摆放棋盘格；实际距离可调整，但必须覆盖足够的深度范围。每个距离执行：

1. 让棋盘格基本正对相机，允许少量倾斜；等待角点识别成功。
2. 根据窗口中的绿色 `Board center / aim laser here` 十字，用云台手动控制，让**真实红色激光点中心**落到棋盘物理中心。
3. 等待云台完全静止。窗口质量行应显示足够大的单格像素尺寸，且 `stable` 达到配置的稳定帧数、`stddev` 不超过阈值。
4. 在标定窗口按空格记录一个样本。终端会输出 `[capture] N point_C=(X,Y,Z)`。
5. 移动棋盘格到下一个距离后再采样；不要在同一距离连续采集多次凑样本数。

按键：

```text
Space      记录当前棋盘中心为一个光束采样点
f          拟合全部样本并写入 JSON
Backspace  删除最近一个样本
r          清空全部样本并重新开始
q / Esc    退出
```

### 拟合、写入与验证

至少采集 `minimum_samples` 个不同距离的样本，且总深度跨度不小于 `minimum_depth_span_m`，再按 `f`。默认配置通常要求至少 6 个样本、2 m 深度跨度和不超过 15 mm 的 RMS 误差。

拟合成功后，程序输出 `test_output/laser_beam_calibration.json`。将其中的：

```json
"beam_line_in_camera": {
  "enabled": 1,
  "point_m": [...],
  "direction": [...]
}
```

覆盖到 `config/laser.json` 的 `laser.beam_line_in_camera`，并确认 `enabled` 为 `1`；否则 `laser_aim` 仍会使用旧的平行光轴回退模型。

随后将棋盘格移到**未参与拟合**的两个距离：不要再手动把红点调到棋盘中心，直接观察窗口中的红色预测十字是否与真实激光点重合。不同距离都能重合后，再运行不发送命令的主程序：

```bash
./build/laser/laser_aim config/laser.json
```

确认目标中心、PnP 距离和 yaw / pitch 修正稳定后，才允许使用：

```bash
./build/laser/laser_aim config/laser.json --send
```

如果拟合失败，优先检查并重采可疑样本，不要直接放宽 `maximum_rms_error_m`。误差随距离持续增大时，重点检查远距离样本、棋盘实际尺寸和相机内参；所有距离向同一方向偏移时，重点检查配置是否正确写入。
