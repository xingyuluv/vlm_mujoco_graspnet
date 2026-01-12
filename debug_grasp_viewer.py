import os
import sys
import cv2
import numpy as np
import open3d as o3d
import torch
import time
import gc

# ================= 路径配置 =================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MANIPULATOR_PATH = os.path.join(PROJECT_ROOT, 'manipulator_grasp')
if MANIPULATOR_PATH not in sys.path:
    sys.path.append(MANIPULATOR_PATH)

GRASPNET_ROOT = os.path.join(PROJECT_ROOT, 'graspnet-baseline')
sys.path.append(os.path.join(GRASPNET_ROOT, 'models'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'dataset'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'utils'))

try:
    from manipulator_grasp.env.ur5_grasp_old_env import UR5GraspEnv
    # 【修改】从 vlm_process 导入 YOLO 和 SAM 相关函数
    from vlm_process import get_yolo_model, get_sam_predictor, process_sam_results
    from grasp_process_old import get_net, get_and_process_data
    from graspnetAPI import GraspGroup
    from graspnet import pred_decode
    from collision_detector import ModelFreeCollisionDetector
    print("✅ 成功加载环境与算法模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 全局变量
is_processing = False

def get_camera_intrinsic(width, height):
    fovy = 45 
    f = 0.5 * height / np.tan(fovy * np.pi / 360)
    return o3d.camera.PinholeCameraIntrinsic(width, height, f, f, width / 2, height / 2)

def create_point_cloud(rgb, depth, intrinsics):
    """
    生成用于显示的点云
    """
    depth_o3d = o3d.geometry.Image(depth)
    color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    
    # 创建 RGBD 图
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, 
        depth_scale=1.0,  # MuJoCo 输出单位是米，这里保持 1.0
        depth_trunc=2.0, 
        convert_rgb_to_intensity=False
    )
    
    # 生成点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    return pcd

def inference_full(net, rgb, depth, mask):
    # 1. 数据预处理
    end_points, cloud_o3d = get_and_process_data(rgb, depth, mask)

    # 2. 前向推理 (无梯度)
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)

    # 3. 构造结果
    preds_np = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(preds_np)

    # 4. 清理中间显存
    del end_points, grasp_preds
    
    # 5. 碰撞检测
    collision_thresh = 0.01
    voxel_size = 0.01
    mfcdetector = ModelFreeCollisionDetector(np.asarray(cloud_o3d.points), voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]

    # 6. NMS 和排序
    gg = gg.nms()
    gg = gg.sort_by_score()

    # 7. 垂直角度过滤
    all_grasps = list(gg)
    vertical = np.array([0, 0, 1]) 
    angle_threshold = np.deg2rad(30) 
    
    filtered_list = []
    for grasp in all_grasps:
        approach_dir = grasp.rotation_matrix[:, 0]
        cos_angle = np.dot(approach_dir, vertical)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            filtered_list.append(grasp)
            
    if len(filtered_list) == 0:
        final_gg = gg
    else:
        print(f"[Info] 垂直过滤: {len(all_grasps)} -> {len(filtered_list)}")
        grasp_array = np.array([g.grasp_array for g in filtered_list])
        final_gg = GraspGroup(grasp_array)

    return final_gg

def main():
    global is_processing
    
    # 清理环境
    gc.collect()
    torch.cuda.empty_cache()
    
    print("🚀 正在启动 MuJoCo 环境...")
    env = UR5GraspEnv()
    env.reset()
    
    # ================= 关键修复开始 =================
    # 定义一个“保持动作”，让机械臂停在初始位置，不要倒下来挡镜头
    home_action = np.zeros(7)
    # 获取 UR5GraspEnv reset 后的默认关节角 (通常是上方横置状态)
    home_action[:6] = env.robot_q 
    # ================= 关键修复结束 =================
    
    print("⏳ 物理预热 (200步)...")
    for _ in range(200): 
        env.step(home_action) # <--- 传入动作
    
    print("🔄 加载模型 (YOLO-World + SAM + GraspNet)...")
    yolo_model = get_yolo_model()
    sam_predictor = get_sam_predictor()
    grasp_net = get_net() 
    print("✅ 模型就绪")

    window_name = "MuJoCo View"
    cv2.namedWindow(window_name)
    
    print("\n" + "="*50)
    print("🎮 操作指南:")
    print("1. 按键盘 'f' 键 -> 输入物体名称进行检测")
    print("2. 按键盘 'q' 键 -> 退出")
    print("3. 在 Open3D 窗口中，用鼠标旋转视角查看抓取位姿")
    print("="*50 + "\n")

    while True:
        # ================= 关键修复 =================
        env.step(home_action) # <--- 持续传入动作，固定机械臂
        # ===========================================
        
        obs = env.render()
        rgb = cv2.cvtColor(obs['img'], cv2.COLOR_RGB2BGR)
        depth = obs['depth']
        
        vis_img = rgb.copy()
        cv2.putText(vis_img, "Press 'f' to Find Object", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, vis_img)
        key = cv2.waitKey(10)
        
        if key == ord('q'): 
            break
        
        if key == ord('f') and not is_processing:
            is_processing = True
            print("\n📝 请输入目标物体名称 (例如: apple, box)")
            try:
                text_prompt = input("👉 请输入: ").strip()
            except EOFError:
                text_prompt = ""
            
            if not text_prompt:
                print("⚠️ 输入为空，取消操作。")
                is_processing = False
                continue

            print(f"🔍 YOLO-World 正在搜索: '{text_prompt}' ...")
            
            try:
                yolo_model.set_classes([text_prompt])
                with torch.no_grad():
                    results = yolo_model.predict(rgb, conf=0.05, iou=0.5, verbose=False)
                
                bbox = None
                if len(results) > 0 and len(results[0].boxes) > 0:
                    best_box = results[0].boxes[0]
                    coords = best_box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(best_box.conf)
                    bbox = coords.tolist()
                    print(f"✅ 找到目标! 置信度: {conf:.2f}, BBox: {bbox}")
                    
                    cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.imshow(window_name, vis_img)
                    cv2.waitKey(10)
                else:
                    print(f"❌ 未找到目标: '{text_prompt}'")
                    is_processing = False
                    continue

                print("🔄 启动 SAM 分割...")
                image_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                with torch.no_grad():
                    sam_predictor.set_image(image_rgb)
                    sam_results = sam_predictor(bboxes=[bbox], points=None, labels=None)
                
                _, mask = process_sam_results(sam_results)
                del sam_results
                
                if mask is not None:
                    mask_vis = cv2.addWeighted(rgb, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
                    cv2.imshow("Mask Result", mask_vis)
                    cv2.waitKey(100)
                    
                    print("🤖 计算抓取点...")
                    # 这里的 inference_full 会调用 GraspNet
                    # 由于机械臂现在不再遮挡，生成的抓取点应该会暴增到 300+
                    gg = inference_full(grasp_net, rgb, depth, mask)
                    
                    if len(gg) > 0:
                        gg = gg.sort_by_score()
                        gg_vis = gg[:50] 
                        print(f"✅ 显示 Top-{len(gg_vis)} 个抓取点")
                        
                        h, w = rgb.shape[:2]
                        intrinsics = get_camera_intrinsic(w, h)
                        cloud = create_point_cloud(rgb, depth, intrinsics)
                        grippers = gg_vis.to_open3d_geometry_list()
                        for gripper in grippers:
                            gripper.paint_uniform_color([1, 0, 0])

                        o3d.visualization.draw_geometries([cloud, *grippers], 
                                                          window_name=f"Detection: {text_prompt} | Top {len(gg_vis)} Grasps")
                    else:
                        print("❌ 未找到有效抓取")
                    del gg
                else:
                    print("❌ SAM 分割失败")

            except Exception as e:
                print(f"❌ 错误: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                is_processing = False
                gc.collect()
                torch.cuda.empty_cache()
                print("♻️  就绪。\n")

    env.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()