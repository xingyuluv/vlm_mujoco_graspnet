import os
import sys
import numpy as np
import torch
import open3d as o3d
from PIL import Image
import spatialmath as sm

from manipulator_grasp.arm.motion_planning import *

from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))
from scipy.spatial.transform import Rotation as R_scipy

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image


# ==================== 网络加载 ====================
def get_net():
    """
    加载训练好的 GraspNet 模型
    """
    net = GraspNet(input_feature_dim=0, 
                   num_view=300, 
                   num_angle=12, 
                   num_depth=4,
                   cylinder_radius=0.05, 
                   hmin=-0.02, 
                   hmax_list=[0.01, 0.02, 0.03, 0.04], 
                   is_training=False)
    net.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接权重的绝对路径
    checkpoint_path = os.path.join(curr_dir, 'logs/log_rs/checkpoint-rs.tar')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ 找不到 GraspNet 权重文件: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path) 
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net




# ================= 数据处理并生成输入 ====================
def get_and_process_data(color_path, depth_path, mask_path):
    """
    根据给定的 RGB 图、深度图、掩码图（可以是 文件路径 或 NumPy 数组），生成输入点云及其它必要数据
    """
#---------------------------------------
    # 1. 加载 color（可能是路径，也可能是数组）
    if isinstance(color_path, str):
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    elif isinstance(color_path, np.ndarray):
        color = color_path.astype(np.float32)
        color /= 255.0
    else:
        raise TypeError("color_path 既不是字符串路径也不是 NumPy 数组！")

    # 2. 加载 depth（可能是路径，也可能是数组）
    if isinstance(depth_path, str):
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img)
    elif isinstance(depth_path, np.ndarray):
        depth = depth_path
    else:
        raise TypeError("depth_path 既不是字符串路径也不是 NumPy 数组！")

    # 3. 加载 mask（可能是路径，也可能是数组）
    if isinstance(mask_path, str):
        workspace_mask = np.array(Image.open(mask_path))
    elif isinstance(mask_path, np.ndarray):
        workspace_mask = mask_path
    else:
        raise TypeError("mask_path 既不是字符串路径也不是 NumPy 数组！")

    # print("\n=== 尺寸验证 ===")
    # print("深度图尺寸:", depth.shape)
    # print("颜色图尺寸:", color.shape[:2])
    # print("工作空间尺寸:", workspace_mask.shape)

    # 构造相机内参矩阵
    height = color.shape[0]
    width = color.shape[1]
    fovy = np.pi / 4 # 定义的仿真相机
    focal = height / (2.0 * np.tan(fovy / 2.0))  # 焦距计算（基于垂直视场角fovy和高度height）
    c_x = width / 2.0   # 水平中心
    c_y = height / 2.0  # 垂直中心
    intrinsic = np.array([
        [focal, 0.0, c_x],    
        [0.0, focal, c_y],   
        [0.0, 0.0, 1.0]
    ])
    factor_depth = 1.0  # 深度因子，根据实际数据调整

    # 利用深度图生成点云 (H,W,3) 并保留组织结构
    camera = CameraInfo(width, height, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # mask = depth < 2.0
    mask = (workspace_mask > 0) & (depth < 2.0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    # print(f"mask过滤后的点云数量 (color_masked): {len(color_masked)}") # 在采样前打印原始过滤后的点数

    NUM_POINT = 10000 # 10000或5000
    # 如果点数足够，随机采样NUM_POINT个点（不重复）
    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    # 如果点数不足，先保留所有点，再随机重复补足NUM_POINT个点
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs] # 提取点云和颜色

    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    # end_points = {'point_clouds': cloud_sampled}

    end_points = dict()
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud_o3d

def refine_grasp_pose(grasp, cloud_o3d, iterations=2):
    """
    [算法创新点] 基于局部点云几何特征的抓取位姿精炼算法 (Refinement)
    
    原理：
    利用 GraspNet 初步预测的位姿裁剪出局部点云，
    通过 PCA (主成分分析) 计算局部几何中心和主轴方向，
    迭代优化抓取位姿，使其更贴合物体表面几何特性。
    
    参数:
        grasp: GraspNet 的单个 Grasp 对象
        cloud_o3d: 场景的 Open3D 点云
        iterations: 迭代次数
        
    返回:
        refined_grasp: 优化后的 Grasp 对象
    """
    # 1. 获取原始参数
    # GraspNet 定义: X=接近方向, Y=闭合方向(两指连线), Z=垂直方向
    R_curr = grasp.rotation_matrix
    t_curr = grasp.translation
    width = grasp.width
    
    # 定义感兴趣区域 (ROI) 的尺寸 (比实际抓取区域稍大一点)
    # 深度(X): 0~0.05m, 宽度(Y): -width/2 ~ width/2, 高度(Z): -0.02 ~ 0.02
    roi_depth = 0.04
    roi_height = 0.03 # 2cm
    
    points_global = np.asarray(cloud_o3d.points)
    
    for i in range(iterations):
        # --- A. 坐标变换: 世界系 -> 当前抓取系 ---
        # T_g_w: World to Grasp
        R_inv = R_curr.T
        t_inv = -R_inv @ t_curr
        
        points_local = (points_global @ R_inv.T) + t_inv
        
        # --- B. 裁剪 ROI (Region of Interest) ---
        # 筛选条件: 
        # X轴 (深度): 在手掌前方 0 到 roi_depth 之间
        # Y轴 (宽度): 在两指之间
        # Z轴 (高度): 在手指厚度范围内
        mask = (points_local[:, 0] > 0.005) & (points_local[:, 0] < roi_depth) & \
               (np.abs(points_local[:, 1]) < width / 2 * 1.2) & \
               (np.abs(points_local[:, 2]) < roi_height)
               
        local_pts = points_local[mask]
        
        # 如果局部点太少，说明抓空了或者噪点，放弃优化，返回原值
        if len(local_pts) < 20:
            # print(f"  [Refine] Iter {i}: Not enough points ({len(local_pts)}), skipping.")
            break
            
        # --- C. 计算局部质心 (Centroid) ---
        # 我们希望夹爪中心对准物体局部质心
        local_centroid = np.mean(local_pts, axis=0)
        
        # --- D. 计算主成分 (PCA) ---
        # 协方差矩阵
        cov_matrix = np.cov(local_pts.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 排序特征向量: eigenvectors[:, 0] 是最小主成分(法向量), [:, 2] 是最大主成分(长轴)
        # 对应关系:
        # 最小主成分 (Normal) -> 应该对齐夹爪 X 轴 (接近方向)
        # 最大主成分 (Major)  -> 应该对齐夹爪 Y 轴 (闭合方向, 或者是Z轴，取决于物体形状)
        # 这里我们主要优化 平移 和 接近方向
        
        # 1. 优化平移 (Translation Refinement)
        # 计算质心在世界坐标系下的位置
        # 注意：我们只修正 Y(左右) 和 Z(上下) 的偏移，保留 X(深度) 不变，防止撞击
        shift_local = np.array([0, local_centroid[1], local_centroid[2]])
        shift_global = R_curr @ shift_local
        t_new = t_curr + shift_global * 0.2 # 0.2 是学习率/阻尼系数，防止震荡
        
        # 2. 优化旋转 (Rotation Refinement) - 可选，比较激进
        # 简单策略：仅微调接近方向，使其更垂直于物体表面 (即平行于法向量)
        # 最小特征向量 v_min 大概是物体表面法向量
        # v_min = eigenvectors[:, 0]
        # if v_min[0] < 0: v_min = -v_min # 确保指向夹爪内部
        
        # 这里为了稳妥，我们只做【位置精炼 (Centering)】，不做旋转，因为旋转容易导致碰撞
        # 如果你想做旋转优化，可以解算 Rotation between X-axis and v_min
        
        # 更新状态
        t_curr = t_new
        # R_curr = ... (如果做了旋转优化)
        
    # 更新 Grasp 对象
    grasp.translation = t_curr
    # grasp.rotation_matrix = R_curr
    
    return grasp

# ==================== 主函数：获取抓取预测 ====================
def run_grasp_inference(color_path, depth_path, sam_mask_path=None):
    # 1. 加载网络
    net = get_net()

    # 2. 处理数据
    end_points, cloud_o3d = get_and_process_data(color_path, depth_path, sam_mask_path)

    # 3. 前向推理
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)

    # 4. 构造 GraspGroup
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

    # 5. 碰撞检测
    COLLISION_THRESH = 0.01
    if COLLISION_THRESH > 0:
        voxel_size = 0.01
        collision_thresh = 0.01
        mfcdetector = ModelFreeCollisionDetector(np.asarray(cloud_o3d.points), voxel_size=voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        gg = gg[~collision_mask]

    # 6. NMS 去重
    gg.nms().sort_by_score()

    # =================【关键新增】坐标系定义 =================
    # 为了判断高度，我们需要先把相机坐标转为世界坐标
    # 这些参数需与 execute_grasp 和 scene.xml 保持一致
    n_wc = np.array([0.0, -1.0, 0.0]) 
    o_wc = np.array([-1.0, 0.0, -0.5]) 
    t_wc = np.array([0.85, 0.8, 1.6]) 
    
    # 构造变换矩阵 T_wc (Camera -> World)
    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    
    # 设定最低安全高度 (世界坐标系)
    # 桌面高度是 0.74m，为了防止抓取过于贴地导致碰撞，设置一个硬阈值
    MIN_SAFE_Z_WORLD = 0.740 + 0.005  # 5cm 安全高度缓冲
    # ========================================================

    all_grasps = list(gg)
    vertical = np.array([0, 0, 1])
    angle_threshold = np.deg2rad(28) 
    
    filtered = []
    
    # 7. 综合过滤循环 (角度 + 高度)
    for grasp in all_grasps:
        # --- A. 垂直角度过滤 ---
        approach_dir = grasp.rotation_matrix[:, 0]
        cos_angle = np.dot(approach_dir, vertical)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # --- B. 【新增】绝对高度过滤 ---
        # 获取相机坐标系下的位置
        t_camera = grasp.translation 
        # 转换到世界坐标系: T_world = T_wc * T_camera_point
        # 注意: spatialmath 的 SE3 * vector 会自动处理齐次变换
        t_world = T_wc * t_camera 
        
        # 只有当 [角度合适] 且 [高度安全] 时才保留
        if angle < angle_threshold and t_world[2] >= MIN_SAFE_Z_WORLD:
            filtered.append(grasp)
            
    if len(filtered) == 0:
        print("\n[Warning] 所有抓取点均因角度或高度限制被过滤，使用原始结果兜底。")
        filtered = all_grasps
    else:
        print(f"\n[DEBUG] 过滤后剩余: {len(filtered)} / {len(all_grasps)} (高度阈值: {MIN_SAFE_Z_WORLD}m)")

    # ===== 计算物体中心点 =====
    points = np.asarray(cloud_o3d.points)
    object_center = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)

    distances = []
    for grasp in filtered:
        grasp_center = grasp.translation
        distance = np.linalg.norm(grasp_center - object_center)
        distances.append(distance)

    grasp_with_distances = [(g, d) for g, d in zip(filtered, distances)]
    
    # ===== 综合得分 =====
    max_distance = max(distances) if distances else 1.0
    grasp_with_composite_scores = []
    
    WEIGHT_SCORE = 0.2
    WEIGHT_DIST = 0.5
    WEIGHT_HORIZONTAL = 0.8

    for g, d in grasp_with_distances:
        distance_score = 1 - (d / max_distance)
        finger_direction_z = g.rotation_matrix[2, 1] 
        horizontal_score = 1.0 - abs(finger_direction_z)
        
        composite_score = (g.score * WEIGHT_SCORE + 
                           distance_score * WEIGHT_DIST + 
                           horizontal_score * WEIGHT_HORIZONTAL)
        grasp_with_composite_scores.append((g, composite_score))

    grasp_with_composite_scores.sort(key=lambda x: x[1], reverse=True)

    if len(grasp_with_composite_scores) > 0:
        filtered = [g for g, score in grasp_with_composite_scores]
        best_z_proj = abs(filtered[0].rotation_matrix[2, 1])
        print(f"\n[DEBUG] Top Grasp Horizontal Score: {1.0 - best_z_proj:.4f}")
    else:
        filtered = [g for g, d in grasp_with_distances]
        filtered.sort(key=lambda g: g.score, reverse=True)

    # 取第1个抓取
    top_grasps = filtered[:1]

    if len(top_grasps) > 0:
        best_grasp = top_grasps[0]
        
        # 8. 局部几何精炼 (Refinement)
        print(f"[*] 正在运行局部几何精炼 (Refinement)...")
        old_t = best_grasp.translation.copy()
        
        # 确保 refine_grasp_pose 函数已定义
        best_grasp = refine_grasp_pose(best_grasp, cloud_o3d, iterations=5)
        
        # 【二次检查】Refinement 可能会把点往下拉，这里再做一次安全检查
        # 转换精炼后的点到世界坐标
        t_world_refined = T_wc * best_grasp.translation
        if t_world_refined[2] < MIN_SAFE_Z_WORLD:
            print(f"⚠️ Refinement 后高度过低 ({t_world_refined[2]:.4f}m)，强制修正回安全高度。")
            # 在世界坐标系下修正 Z
            t_world_refined[2] = MIN_SAFE_Z_WORLD
            # 逆变换回相机坐标系赋值给 grasp (T_cam = inv(T_wc) * T_world)
            best_grasp.translation = (T_wc.inv() * t_world_refined)

        diff = np.linalg.norm(best_grasp.translation - old_t)
        print(f"[*] 精炼完成: 位置修正量 = {diff*1000:.2f} mm")
        
        new_gg = GraspGroup()
        new_gg.add(best_grasp)

        visual = True
        if visual:
            grippers = new_gg.to_open3d_geometry_list()
            for gripper in grippers:  
                gripper.paint_uniform_color([1, 0, 0])
            o3d.visualization.draw_geometries([cloud_o3d, *grippers], window_name="Top Grasp Visualization")
        
        return new_gg
    else:
        print("[Error] No valid grasp found after processing.")
        return GraspGroup()


# ================= 仿真执行抓取动作 ====================
def execute_grasp(env, gg):
    """
    执行抓取动作，控制机器人从初始位置移动到抓取位置，并完成抓取操作。

    参数:
    env (UR5GraspEnv): 机器人环境对象。
    gg (GraspGroup): 抓取预测结果。
    """
    robot = env.robot
    T_wb = robot.base

    # 0.初始准备阶段
    # 目标：计算抓取位姿 T_wo（物体相对于世界坐标系的位姿）
    # n_wc = np.array([0.0, -1.0, 0.0]) # 相机朝向
    # o_wc = np.array([-1.0, 0.0, -0.5]) # 相机朝向 [0.5, 0.0, -1.0] -> [-1.0, 0.0, -0.5]
    # t_wc = np.array([1.0, 0.6, 2.0]) # 相机的位置。2.0是相机高度，与scene.xml中保持一致。
    n_wc = np.array([0.0, -1.0, 0.0]) 
    o_wc = np.array([-1.0, 0.0, -0.5]) 
    t_wc = np.array([0.85, 0.8, 1.6]) 

    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))
    T_wo = T_wc * T_co
    Z_HEIGHT_OFFSET = 0.02 
    T_wo = sm.SE3.Trans(0, 0, Z_HEIGHT_OFFSET) * T_wo

    action = np.zeros(7)

    # 1.机器人运动到预抓取位姿
    # 目标：将机器人从当前位置移动到预抓取姿态（q1）
    time1 = 1
    q0 = robot.get_joint()
    q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
    parameter0 = JointParameter(q0, q1)
    velocity_parameter0 = QuinticVelocityParameter(time1)
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)
    planner1 = TrajectoryPlanner(trajectory_parameter0)
    # 执行planner_array = [planner1]
    time_array = [0.0, time1]
    planner_array = [planner1]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

    # 2.接近抓取位姿
    # 目标：从预抓取位姿直线移动到抓取点附近（T2）
    # 关键点：T2 是 T_wo 沿负 x 方向偏移 0.1m，确保安全接近物体。
    time2 = 1
    robot.set_joint(q1)
    T1 = robot.get_cartesian()
    T2 = T_wo * sm.SE3(-0.1, 0.0, 0.0)
    position_parameter1 = LinePositionParameter(T1.t, T2.t) #  位置规划（直线路径）
    attitude_parameter1 = OneAttitudeParameter(sm.SO3(T1.R), sm.SO3(T2.R)) # 姿态规划（插值旋转）
    cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1) # 组合笛卡尔参数
    velocity_parameter1 = QuinticVelocityParameter(time2) # 速度曲线（五次多项式插值）
    trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1) # 将笛卡尔空间路径和速度曲线结合，生成完整的轨迹参数
    planner2 = TrajectoryPlanner(trajectory_parameter1) # 轨迹规划器，将笛卡尔空间路径和速度曲线结合，生成完整的轨迹参数
    # 执行planner_array = [planner2]
    time_array = [0.0, time2]
    planner_array = [planner2]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

    # 3.执行抓取
    # 目标：从 T2 移动到 T3（精确抓取位姿）。通过逐步增加 action[-1]（夹爪控制信号）闭合夹爪，抓取物体。
# 3.执行抓取
    # 目标：从 T2 移动到 T3（精确抓取位姿）。
    # 【修改点】让 T3 比 T_wo 更深一点 (沿末端 X 轴前进)
    time3 = 1.5  # 建议稍微把时间改长一点 (原为1)，动作更稳
    
    # 设定插入深度，例如 1.5cm (0.015m) 到 2cm (0.02m)
    # 如果物体较小，0.015 比较安全；如果没抓到，可以加到 0.025
    INSERT_DEPTH = 0.015 
    
    # T_wo 是刚才计算好的抓取点(已包含高度修正)
    # 乘以 sm.SE3(INSERT_DEPTH, 0, 0) 表示沿当前末端的 X 轴(向前)移动
    T3 = T_wo * sm.SE3(INSERT_DEPTH, 0.0, 0.0)
    
    position_parameter2 = LinePositionParameter(T2.t, T3.t)
    attitude_parameter2 = OneAttitudeParameter(sm.SO3(T2.R), sm.SO3(T3.R))
    cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
    velocity_parameter2 = QuinticVelocityParameter(time3)
    trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
    planner3 = TrajectoryPlanner(trajectory_parameter2)
    # 执行planner_array = [planner3]
    time_array = [0.0, time3]
    planner_array = [planner3]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num) 
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)): 
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break
    for i in range(1000):
        action[-1] += 0.2
        action[-1] = np.min([action[-1], 255])
        env.step(action)

    # 4.提起物体
    # 目标：抓取后垂直提升物体（避免碰撞桌面）。
    time4 = 1
    T4 = sm.SE3.Trans(0.0, 0.0, 0.3) * T3 # 通过在T3的基础上向上偏移0.3单位得到的，用于控制机器人上升一定的高度
    position_parameter3 = LinePositionParameter(T3.t, T4.t)
    attitude_parameter3 = OneAttitudeParameter(sm.SO3(T3.R), sm.SO3(T4.R))
    cartesian_parameter3 = CartesianParameter(position_parameter3, attitude_parameter3)
    velocity_parameter3 = QuinticVelocityParameter(time4)
    trajectory_parameter3 = TrajectoryParameter(cartesian_parameter3, velocity_parameter3)
    planner4 = TrajectoryPlanner(trajectory_parameter3)

    # 5.水平移动物体
    # 目标：将物体水平移动到目标放置位置，保持高度不变。
    time5 = 1
    T5 = sm.SE3.Trans(1.4, 0.3, T4.t[2]) * sm.SE3(sm.SO3(T4.R)) #  通过在T4的基础上进行平移得到，这里的1.4, 0.3是场景中的固定点坐标，而不是偏移量
    position_parameter4 = LinePositionParameter(T4.t, T5.t)
    attitude_parameter4 = OneAttitudeParameter(sm.SO3(T4.R), sm.SO3(T5.R))
    cartesian_parameter4 = CartesianParameter(position_parameter4, attitude_parameter4)
    velocity_parameter4 = QuinticVelocityParameter(time5)
    trajectory_parameter4 = TrajectoryParameter(cartesian_parameter4, velocity_parameter4)
    planner5 = TrajectoryPlanner(trajectory_parameter4)

    # 6.放置物体
    # 目标：垂直下降物体到接触面（T7）。逐步减小 action[-1]（夹爪信号）以释放物体。
    time6 = 1
    T6 = sm.SE3.Trans(0.0, 0.0, -0.1) * T5 # 通过在T5的基础上向下偏移0.1单位得到的，用于控制机器人下降一定的高度
    position_parameter6 = LinePositionParameter(T5.t, T6.t)
    attitude_parameter6 = OneAttitudeParameter(sm.SO3(T5.R), sm.SO3(T6.R))
    cartesian_parameter6 = CartesianParameter(position_parameter6, attitude_parameter6)
    velocity_parameter6 = QuinticVelocityParameter(time6)
    trajectory_parameter6 = TrajectoryParameter(cartesian_parameter6, velocity_parameter6)
    planner6 = TrajectoryPlanner(trajectory_parameter6)

    # 执行planner_array = [planner4, planner5, planner6]
    time_array = [0.0, time4, time5, time6]
    planner_array = [planner4, planner5, planner6]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break
    for i in range(1000):
        action[-1] -= 0.2
        action[-1] = np.max([action[-1], 0])
        env.step(action)

    # 7.抬起夹爪
    # 目标：放置后抬起夹爪，避免碰撞物体。
    time7 = 1
    T7 = sm.SE3.Trans(0.0, 0.0, 0.1) * T6
    position_parameter7 = LinePositionParameter(T6.t, T7.t)
    attitude_parameter7 = OneAttitudeParameter(sm.SO3(T6.R), sm.SO3(T7.R))
    cartesian_parameter7 = CartesianParameter(position_parameter7, attitude_parameter7)
    velocity_parameter7 = QuinticVelocityParameter(time7)
    trajectory_parameter7 = TrajectoryParameter(cartesian_parameter7, velocity_parameter7)
    planner7 = TrajectoryPlanner(trajectory_parameter7)
    # 执行planner_array = [planner7]
    time_array = [0.0, time7]
    planner_array = [planner7]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

    # 8.回到初始位置
    # 目标：机器人返回初始姿态（q0），完成整个任务。
    time8 = 1
    q8 = robot.get_joint()
    q9 = q0
    parameter8 = JointParameter(q8, q9)
    velocity_parameter8 = QuinticVelocityParameter(time8)
    trajectory_parameter8 = TrajectoryParameter(parameter8, velocity_parameter8)
    planner8 = TrajectoryPlanner(trajectory_parameter8)
    # 执行planner_array = [planner8]
    time_array = [0.0, time8]
    planner_array = [planner8]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break 