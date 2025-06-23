import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Dynamic Ray Caster Test Script")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch
import random
import numpy as np
import time
from pack import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import isaacsim.core.utils.prims as prim_utils
from pxr import Usd, UsdGeom, Gf
import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def convert_transform_to_list(transform, device):
    # Extract location information
    pos = [transform.position.x, transform.position.y, transform.position.z]
    # Convert Euler angles (in degrees) to radians
    roll = math.radians(transform.attitude.roll)
    pitch = math.radians(transform.attitude.pitch)
    yaw = math.radians(transform.attitude.yaw)
    # Convert float to Tensor, specifying data type and device
    roll_tensor = torch.tensor(roll, dtype=torch.float32, device=device)
    pitch_tensor = torch.tensor(pitch, dtype=torch.float32, device=device)
    yaw_tensor = torch.tensor(yaw, dtype=torch.float32, device=device)
    # print(roll_tensor, pitch_tensor, yaw_tensor)
    # Use quat_from_euler_xyz, note that this function returns a Tensor
    quat_tensor = math_utils.quat_from_euler_xyz(roll_tensor, pitch_tensor, yaw_tensor)
    # Convert a quaternion Tensor to a list
    quat_list = quat_tensor.tolist()
    # Concatenate positions and quaternions to form a list of 7 numbers
    result = pos + quat_list
    return result

def getSurfaceItem(zSize, xSize, ySize):
    """
    Generates an Item object with only the surface according to the given size.
    Parameter order: (zSize, xSize, ySize), consistent with the size order of the entire system (z, x, y).
    """
    cube = np.ones((zSize, xSize, ySize))
    if zSize > 2 and xSize > 2 and ySize > 2:
        cube[1: zSize-1, 1: xSize-1, 1: ySize-1] = 0
    return Item(cube)

def get_suitcase_size(usd_path):
    # 替换为你的 USD 文件路径
    stage = Usd.Stage.Open(usd_path)

    # 替换为你想要查询的 prim 路径
    prim = stage.GetDefaultPrim()
    if not prim.IsValid():
        print("Invalid prim")
        return

    # 创建 BBoxCache 实例
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
        ignoreVisibility=False
    )

    # 计算世界空间的 bounding box
    bbox = bbox_cache.ComputeWorldBound(prim)

    # 获取轴对齐的包围盒
    aligned_box = bbox.ComputeAlignedBox()

    # 获取最小点和最大点
    min_point = aligned_box.GetMin()
    max_point = aligned_box.GetMax()

    # 计算尺寸
    # print(f"min_point: {min_point[0], min_point[1], min_point[2]}, max_point: {max_point[0], max_point[1], max_point[2]}")
    size_x = round((max_point[0] - min_point[0]))
    size_y = round((max_point[1] - min_point[1]))
    size_z = round((max_point[2] - min_point[2]))
    suitcase_size = []
    suitcase_size.append(size_x)
    suitcase_size.append(size_y)
    suitcase_size.append(size_z)
    # print(f"尺寸 (x, y, z): ({suitcase_size})")

    return suitcase_size

def crop_valid_region(ray_hits_w: torch.Tensor) -> torch.Tensor:
    """
    过滤掉 ray_hits_w 中 z == 0 或任意维度为 inf 的无效点。

    输入:
        ray_hits_w: Tensor, shape (N, B, 3)

    输出:
        filtered: Tensor, shape (N, M, 3)，只包含有效点
    """
    assert ray_hits_w.dim() == 3 and ray_hits_w.shape[2] == 3, "输入必须是 (N, B, 3)"

    # 去除 inf 点：对每个 ray，如果任意维度是 inf，就无效
    inf_mask = torch.isinf(ray_hits_w).any(dim=2)  # shape: (N, B)

    # 去除 z == 0 的点
    z_zero_mask = ray_hits_w[:, :, 2] == 0  # shape: (N, B)

    # 合并无效条件
    invalid_mask = inf_mask | z_zero_mask

    # 有效点的 mask
    valid_mask = ~invalid_mask  # shape: (N, B)

    # 对每个传感器单独处理（适用于 N > 1）
    filtered = []
    for i in range(ray_hits_w.shape[0]):
        valid_points = ray_hits_w[i][valid_mask[i]]
        filtered.append(valid_points.unsqueeze(0))  # shape: [1, M_i, 3]

    # 拼接为一个 batch
    return torch.cat(filtered, dim=0)  # shape: [N, M, 3]

def get_bounding_box_size(heightmap_topdown: torch.Tensor, heightmap_bottomup: torch.Tensor):
    """
    从 topdown 和 bottomup 点云中计算 x, y, z 三个方向的范围尺寸。

    参数:
        heightmap_topdown: Tensor (1, N1, 3)
        heightmap_bottomup: Tensor (1, N2, 3)

    返回:
        x_size, y_size, z_size: 每个方向的 float 尺寸
    """
    # 取出有效点
    top = crop_valid_region(heightmap_topdown)[0]      # shape: [N1', 3]
    bottom = crop_valid_region(heightmap_bottomup)[0]  # shape: [N2', 3]

    # 分别计算范围
    x_min, x_max = bottom[:, 0].min(), bottom[:, 0].max()
    y_min, y_max = bottom[:, 1].min(), bottom[:, 1].max()
    z_min = bottom[:, 2].min()
    z_max = top[:, 2].max()

    x_size = round((x_max - x_min).item(),1)
    y_size = round((y_max - y_min).item(),1)
    z_size = round((z_max - z_min).item(),1)

    return x_size, y_size, z_size

def design_scene() -> dict:
    """Design the scene."""
    # # -- Rough terrain
    # cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd")
    # cfg.func("/World/Ground", cfg)

    # Ground-plane
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/flat_plane.usd")
    cfg.func("/World/Ground", cfg)

    # spawn distant light
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    #box size(z,x,y)
    box_size = (20, 40, 40)
    wall_thickness = 2

    # Create separate groups called "Origin1", "Origin2", "Origin3",...(x，y，z)
    # Each group will have a item in it
    origins = [[0, 0, 0], 
               [0, 0, 0],
               [box_size[1]/2, -wall_thickness/2, box_size[0]/2], 
               [box_size[1]/2, box_size[2]+wall_thickness/2, box_size[0]/2], 
               [-wall_thickness/2, box_size[2]/2, box_size[0]/2],
               [box_size[1]+wall_thickness/2, box_size[2]/2, box_size[0]/2],
               [0, 0, 0],
               [0, 0, 0]
               ]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # -- ball
    balls = {}
    ball_cfg_1 = RigidObjectCfg(
        prim_path="/World/Origin0/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(random.random(), random.random(), random.random()), 
                    metallic=0.2
                ),
        ),
    )
    ball_1 = RigidObject(ball_cfg_1)
    balls["ball_1"] = ball_1

    ball_cfg_2 = RigidObjectCfg(
        prim_path="/World/Origin1/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(random.random(), random.random(), random.random()), 
                    metallic=0.2
                ),
        ), 
        init_state=RigidObjectCfg.InitialStateCfg(
                pos=(50, 0, 0),  # Origin1
        ),
    )
    ball_2 = RigidObject(ball_cfg_2)
    balls["ball_2"] = ball_2

    # container
    containers = {}
    container_1_cfg = RigidObjectCfg(
        prim_path=f"/World/Origin2/Container_1",
        spawn=sim_utils.MeshCuboidCfg(
                size=(box_size[1], wall_thickness, box_size[0]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0), 
                    metallic=0.2
                ),
            ),
    )
    container_1 = RigidObject(container_1_cfg)
    containers["container_1"] = container_1

    container_2_cfg = RigidObjectCfg(
        prim_path=f"/World/Origin3/Container_2",
        spawn=sim_utils.MeshCuboidCfg(
                size=(box_size[1], wall_thickness, box_size[0]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0), 
                    metallic=0.2
                ),
            ),
    )
    container_2 = RigidObject(container_2_cfg)
    containers["container_2"] = container_2

    container_3_cfg = RigidObjectCfg(
        prim_path=f"/World/Origin4/Container_3",
        spawn=sim_utils.MeshCuboidCfg(
                size=(wall_thickness, box_size[2], box_size[0]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0), 
                    metallic=0.2
                ),
            ),
    )
    container_3 = RigidObject(container_3_cfg)
    containers["container_3"] = container_3

    container_4_cfg = RigidObjectCfg(
        prim_path=f"/World/Origin5/Container_4",
        spawn=sim_utils.MeshCuboidCfg(
                size=(wall_thickness, box_size[2] , box_size[0]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0), 
                    metallic=0.2
                ),
            ),
    )
    container_4 = RigidObject(container_4_cfg)
    containers["container_4"] = container_4

    # -- suitcases
    suitcases = {}
    data_suitcases = {}
    usd = "/home/yu/IsaacLab/source/isaaclab_assets/manibot/suitcase_large.usd"
    suitcase_cfg = RigidObjectCfg(
        prim_path=f"/World/Origin6/Suitcase",
        spawn=sim_utils.UsdFileCfg(
            usd_path=(usd),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(random.random(), random.random(), random.random()), 
                metallic=0.2
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
                pos=(65, 15, 15),  # Origin6
        ),
    )
    suitcase_1 = RigidObject(cfg=suitcase_cfg)
    suitcases["suitcase_1"] = suitcase_1

    data_suitcase_cfg = RigidObjectCfg(
        prim_path=f"/World/Origin7/Data_suitcase",
        spawn=sim_utils.UsdFileCfg(
            usd_path=(usd),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                rigid_body_enabled=True,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(random.random(), random.random(), random.random()), 
                metallic=0.2
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
                pos=(65, -15, 15),  # Origin7
        ),
    )
    data_suitcase_1 = RigidObject(cfg=data_suitcase_cfg)
    data_suitcases["data_suitcase_1"] = data_suitcase_1

    # Create a ray-caster sensor
    ray_casters = {}
    ray_caster_cfg_1 = RayCasterCfg(
        prim_path="/World/Origin0/ball",
        offset=RayCasterCfg.OffsetCfg(pos=(box_size[1]/2, box_size[2]/2, box_size[0]+5)),
        mesh_prim_paths=["/World/Ground", "/World/Origin.*/Suitcase"],
        pattern_cfg=patterns.GridPatternCfg(resolution=1, size=(box_size[1]-1, box_size[2]-1)),
        attach_yaw_only=True,
        debug_vis=not args_cli.headless,
    )
    ray_caster_1 = RayCaster(cfg=ray_caster_cfg_1)
    ray_casters["ray_caster_1"] = ray_caster_1

    ray_caster_cfg_2 = RayCasterCfg(
        prim_path="/World/Origin1/ball",
        offset=RayCasterCfg.OffsetCfg(pos=(15, -15, 30)),
        mesh_prim_paths=["/World/Ground", "/World/Origin.*/Data_suitcase"],
        pattern_cfg=patterns.GridPatternCfg(resolution=1, size=(30, 30), direction = (0, 0, -1)),
        attach_yaw_only=True,
        debug_vis=not args_cli.headless,
    )
    ray_caster_2 = RayCaster(cfg=ray_caster_cfg_2)
    ray_casters["ray_caster_2"] = ray_caster_2

    ray_caster_cfg_3 = RayCasterCfg(
        prim_path="/World/Origin1/ball",
        offset=RayCasterCfg.OffsetCfg(pos=(15, -15, 0)),
        mesh_prim_paths=["/World/Origin.*/Data_suitcase"],
        pattern_cfg=patterns.GridPatternCfg(resolution=1, size=(30, 30), direction = (0, 0, 1)),
        attach_yaw_only=True,
        debug_vis=not args_cli.headless,
    )
    ray_caster_3 = RayCaster(cfg=ray_caster_cfg_3)
    ray_casters["ray_caster_3"] = ray_caster_3

    # return the scene information
    scene_entities = {"balls": balls, "suitcases": suitcases, "data_suitcases":data_suitcases, "ray_casters": ray_casters, "box_size": box_size, "containers":containers}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    balls: dict = scene_entities["balls"]
    suitcases: dict = scene_entities["suitcases"]
    data_suitcases: dict = scene_entities["data_suitcases"]
    ray_casters: dict = scene_entities["ray_casters"]
    box_size = scene_entities["box_size"]
    containers = scene_entities["containers"]

    # Simulation step counter.
    dt = sim.get_physics_dt()
    count = 0

    # Get a list of all objects
    suitcases_list = list(suitcases.values())
    container_list = list(containers.values())
    balls_list = list(balls.values())
    data_suitcases_list = list(data_suitcases.values())
    ray_casters_list = list(ray_casters.values())

    # --- Packing Algorithm ---
    # Initialize packing problem with box size and items
    items = []
    suitcase_size_large = get_suitcase_size("/home/yu/IsaacLab/source/isaaclab_assets/manibot/suitcase_large.usd")
    for i in range(len(suitcases)):
        item = getSurfaceItem(suitcase_size_large[2], suitcase_size_large[0], suitcase_size_large[1])
        items.append(item)
    current_idx = 0  # The index of the object to be placed

    # 存放所有可能的变换矩阵
    stable_attitudes_score = PriorityQueue()
    # 将容器划分为 grid_num * grid_num 个网格
    # 对于每个网格，尝试放下物体
    grid_coords = []
    grid_num=5
    for i in range(grid_num):
        for j in range(grid_num):
            x = math.floor(box_size[1] * i / grid_num)
            y = math.floor(box_size[2] * j / grid_num)
            grid_coords.append([x, y])

    # Get the default state of the ball 
    ball_default_state_1 = balls["ball_1"].data.default_root_state.clone()
    ball_default_state_2 = balls["ball_2"].data.default_root_state.clone()
    
    # Get the default state of the data_suitcase
    data_suitcase_default_state_1 = data_suitcases["data_suitcase_1"].data.default_root_state.clone()


    while simulation_app.is_running():  
        # If there are still unplaced object, place the next one
        if count == 100 :
            sim_start_time = time.time()
            for roll in range(0, 360, 90):
                for pitch in range(0, 360, 90):
                    for yaw in range(0, 360, 90):
                        # 构造新状态：位置 + 四元数
                        # roll = math.radians(roll)
                        # pitch = math.radians(pitch)
                        yaw = math.radians(yaw)
                        roll_tensor = torch.tensor(0, dtype=torch.float32)
                        pitch_tensor = torch.tensor(0, dtype=torch.float32)
                        yaw_tensor = torch.tensor(yaw, dtype=torch.float32)
                        quat_tensor = math_utils.quat_from_euler_xyz(roll_tensor, pitch_tensor, yaw_tensor).flatten()
                        #更新位置
                        data_suitcase_new_state_1 = data_suitcase_default_state_1.clone()
                        data_suitcase_new_state_1[:,3:7] = quat_tensor
                        data_suitcases["data_suitcase_1"].write_root_pose_to_sim(data_suitcase_new_state_1[:, :7])
                        data_suitcases["data_suitcase_1"].update(dt)
                        ray_casters["ray_caster_2"].update(dt)
                        ray_casters["ray_caster_3"].update(dt)
                        current_idx += 1 
                        step_start_time = time.time()
                        sim.step()
                        count += 1
                        print(f"[INFO]: step time: {time.time() - step_start_time:.2f} seconds")

                        # update buffers
                        update_start_time = time.time()
                        for ray_caster in ray_casters_list:
                            ray_caster.update(dt, force_recompute=True) 
                        for suitcase in suitcases_list:
                            suitcase.update(dt)
                        for data_suitcase in data_suitcases_list:
                            data_suitcase.update(dt)
                        for container in container_list:
                            container.update(dt)
                        for ball in balls_list:
                            ball.update(dt)
                        print(f"[INFO]: update time: {time.time() - update_start_time:.2f} seconds")

                        heightmap_topdown = ray_casters["ray_caster_2"].data.ray_hits_w
                        top = crop_valid_region(heightmap_topdown)
                        heightmap_bottomup = ray_casters["ray_caster_3"].data.ray_hits_w
                        bottom = crop_valid_region(heightmap_bottomup)
                        # print(f"at current_idx:", current_idx, "\n", "heightmap_topdown:", top, "\n", "heightmap_bottomup:", bottom,"\n")
                        x_size, y_size, z_size = get_bounding_box_size(top, bottom)
                        print(f"x_size = {x_size}, y_size = {y_size}, z_size = {z_size}")
                        print(f"suitcase_size_large:", suitcase_size_large)
                        print(f"[INFO]: rotate time: {time.time() - sim_start_time:.2f} seconds")


        # update buffers
        balls["ball_1"].write_root_pose_to_sim(ball_default_state_1[:, :7])
        balls["ball_2"].write_root_pose_to_sim(ball_default_state_2[:, :7])
            
        # Step the simulation.
        sim.step()
        count += 1

        # update buffers
        for ray_caster in ray_casters_list:
            ray_caster.update(dt, force_recompute=True) 
        for suitcase in suitcases_list:
            suitcase.update(dt)
        for data_suitcase in data_suitcases_list:
            data_suitcase.update(dt)
        for container in container_list:
            container.update(dt)
        for ball in balls_list:
            ball.update(dt)


def main():
    """Main function."""
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([70, 70, 80], [0.0, 0.0, 0.0])
    # Design the scene
    scene_entities = design_scene()
    # Play simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run simulator
    run_simulator(sim=sim, scene_entities=scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()