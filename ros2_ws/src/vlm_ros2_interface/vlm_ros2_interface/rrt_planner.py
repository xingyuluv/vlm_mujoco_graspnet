# 文件名: rrt_planner.py
import numpy as np
import mujoco
import time

class Node:
    def __init__(self, q):
        self.q = np.array(q)
        self.parent = None
        self.cost = 0.0

class MujocoCollisionChecker:
    def __init__(self, mj_model, mj_data):
        self.model = mj_model
        # 创建独立的 mjData 避免线程冲突
        self.data = mujoco.MjData(mj_model)
        self.update_world_state(mj_data)

    def update_world_state(self, main_data):
        """同步主环境状态，保持物品位置一致"""
        self.data.qpos[:] = main_data.qpos[:]
        mujoco.mj_forward(self.model, self.data)

    def is_collision(self, q, verbose=False):
        """
        核心修复：高精度碰撞检测 + 智能接触白名单
        """
        self.data.qpos[:6] = q
        
        # 🚨 核心修复 1：必须使用 mj_forward 来更新 BVH 碰撞树，否则会发生穿模！
        mujoco.mj_forward(self.model, self.data)
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or "unknown"
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or "unknown"
            
            robot_keywords = ['ur', 'shoulder', 'arm', 'wrist', 'link', 'flange', '2f85', 'finger', 'pad', 'robotiq', 'base']
            obj_keywords = ['Apple', 'Banana', 'Duck', 'hammer', 'mouse', 'duck']
            
            is_g1_robot = any(k in g1 for k in robot_keywords)
            is_g2_robot = any(k in g2 for k in robot_keywords)
            is_g1_obj = any(k in g1 for k in obj_keywords)
            is_g2_obj = any(k in g2 for k in obj_keywords)
            
            # 🚨 核心修复 2：防误报智能白名单
            # 1. 忽略完全不包含机械臂和物品的接触 (比如: 红色挡板碰到了桌子) -> 解决 Step 8 起点报错
            if not (is_g1_robot or is_g2_robot or is_g1_obj or is_g2_obj):
                continue
            # 2. 忽略机器人内部的自碰撞
            if is_g1_robot and is_g2_robot:
                continue
            # 3. 忽略夹爪和物品的接触 (抓取必然发生)
            if (is_g1_robot and is_g2_obj) or (is_g2_robot and is_g1_obj):
                continue
            # 4. 忽略物品和环境(桌子/地面)的自然放置接触
            if (is_g1_obj and not is_g2_robot and not is_g2_obj):
                continue
            if (is_g2_obj and not is_g1_robot and not is_g1_obj):
                continue
            # 5. 忽略物品与物品的碰撞
            if is_g1_obj and is_g2_obj:
                continue
                
            # 💥 运行到这里，说明真的是机械臂撞到墙壁或桌子了！
            if verbose:
                print(f"💥 碰撞拦截: {g1} <---> {g2}")
            return True
            
        return False

class RRTStarPlanner:
    def __init__(self, collision_checker, joint_limits_min, joint_limits_max):
        self.checker = collision_checker
        self.q_min = np.array(joint_limits_min)
        self.q_max = np.array(joint_limits_max)
        
        # 参数配置
        self.max_iter = 5000      
        self.step_size = 0.3      
        self.search_radius = 0.8  
        self.goal_bias = 0.05

    def plan(self, start_q, goal_q, main_mj_data=None):
        if main_mj_data is not None:
            self.checker.update_world_state(main_mj_data)

        start_time = time.time()
        start_node = Node(start_q)
        goal_node = Node(goal_q)
        self.node_list = [start_node]

        if self.checker.is_collision(start_q, verbose=True):
            print("❌ RRT 错误: 起点处于碰撞状态 (请看上方的 💥 拦截日志)")
            return None
        if self.checker.is_collision(goal_q, verbose=True):
            print("❌ RRT 错误: 终点处于碰撞状态 (目标点可能在物体内部或墙内)")
            return None

        # 放大采样空间，给机械臂留出绕大弯的空间
        margin = 2.0
        self.sample_min = np.maximum(np.minimum(start_q, goal_q) - margin, self.q_min)
        self.sample_max = np.minimum(np.maximum(start_q, goal_q) + margin, self.q_max)

        print(f"🔄 RRT* 开始避障规划...")

        for i in range(self.max_iter):
            if np.random.random() > self.goal_bias: 
                rnd_q = self.random_sample()
            else:
                rnd_q = goal_node.q
            
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_q)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_q, self.step_size)

            if not self.check_segment_collision(nearest_node.q, new_node.q):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                self.node_list.append(new_node)
                self.rewire(new_node, near_inds)

                if not self.check_segment_collision(new_node.q, goal_node.q):
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + np.linalg.norm(new_node.q - goal_node.q)
                    print(f"✅ RRT* 成功跨越障碍! 迭代: {i}, 耗时: {time.time()-start_time:.2f}s")
                    raw_path = self.generate_final_course(goal_node)
                    return self.prune_path(raw_path)

        print(f"⚠️ RRT* 耗尽 {self.max_iter} 迭代未找到路径，耗时: {time.time()-start_time:.2f}s")
        return None

    def prune_path(self, path):
        if len(path) < 3: return path
        pruned_path = [path[0]]
        current_idx = 0
        while current_idx < len(path) - 1:
            for next_idx in range(len(path) - 1, current_idx, -1):
                if next_idx == current_idx + 1:
                    pruned_path.append(path[next_idx])
                    current_idx = next_idx
                    break
                # 平滑时也使用极小步长，严防穿模
                if not self.check_segment_collision(path[current_idx], path[next_idx], step=0.02):
                    pruned_path.append(path[next_idx])
                    current_idx = next_idx
                    break
        return pruned_path

    def random_sample(self):
        return np.random.uniform(self.sample_min, self.sample_max)

    def get_nearest_node_index(self, node_list, rnd_q):
        dlist = [np.linalg.norm(node.q - rnd_q) for node in node_list]
        return dlist.index(min(dlist))

    def steer(self, from_node, to_q, extend_length=float("inf")):
        diff = to_q - from_node.q
        distance = np.linalg.norm(diff)
        if distance > extend_length:
            new_q = from_node.q + diff * (extend_length / distance)
        else:
            new_q = to_q
        new_node = Node(new_q)
        new_node.parent = from_node
        new_node.cost = from_node.cost + np.linalg.norm(new_q - from_node.q)
        return new_node

    # 🚨 核心修复 3：步长设置为 0.02 弧度，绝不放过任何 4cm 的薄墙壁！
    def check_segment_collision(self, q1, q2, step=0.02):
        dist = np.linalg.norm(q2 - q1)
        if dist < step:
            return self.checker.is_collision(q2)
        num_steps = int(dist / step)
        direction = (q2 - q1) / dist
        for i in range(1, num_steps + 1):
            q_check = q1 + direction * (i * step)
            if self.checker.is_collision(q_check):
                return True
        return self.checker.is_collision(q2)

    def find_near_nodes(self, new_node):
        n_node = len(self.node_list) + 1
        r = min(self.search_radius * np.sqrt((np.log(n_node) / n_node)), self.search_radius)
        dist_list = [np.linalg.norm(node.q - new_node.q) for node in self.node_list]
        return [i for i, d in enumerate(dist_list) if d <= r]

    def choose_parent(self, new_node, near_inds):
        if not near_inds: return new_node
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            if not self.check_segment_collision(near_node.q, new_node.q):
                costs.append(near_node.cost + np.linalg.norm(new_node.q - near_node.q))
            else:
                costs.append(float("inf"))
        min_cost = min(costs)
        if min_cost == float("inf"): return new_node
        new_node.parent = self.node_list[near_inds[costs.index(min_cost)]]
        new_node.cost = min_cost
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node.q)
            if edge_node.q is not None:
                edge_cost = np.linalg.norm(near_node.q - new_node.q)
                if new_node.cost + edge_cost < near_node.cost:
                    if not self.check_segment_collision(new_node.q, near_node.q):
                        near_node.parent = new_node
                        near_node.cost = new_node.cost + edge_cost

    def generate_final_course(self, goal_node):
        path = [goal_node.q]
        node = goal_node
        while node.parent is not None:
            node = node.parent
            path.append(node.q)
        path.reverse()
        return path