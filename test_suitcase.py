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
from pack import *

import isaacsim.core.utils.prims as prim_utils
from pxr import Usd, UsdGeom, Gf
import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns



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
    cube[1: zSize-1, 1: xSize-1, 1: ySize-1] = 0
    return Item(cube)

def get_suitcase_size():
    # 替换为你的 USD 文件路径
    stage = Usd.Stage.Open("/home/yu/IsaacLab/source/isaaclab_assets/manibot/suitcase.usd")

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
    origins = [[-5, -5, 0], 
               [box_size[1]/2, -wall_thickness/2, box_size[0]/2], 
               [box_size[1]/2, box_size[2]+wall_thickness/2, box_size[0]/2], 
               [-wall_thickness/2, box_size[2]/2, box_size[0]/2],
               [box_size[1]+wall_thickness/2, box_size[2]/2, box_size[0]/2],
               [50, 50, 0],
               [50, 60, 0],
               [50, 70, 0],
               [50, 80, 0],
               [50, 90, 0],
               [60, 50, 0],
               [60, 60, 0],
               [60, 70, 0],
               [60, 80, 0],
               ]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # -- ball
    cfg = RigidObjectCfg(
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
    ball = RigidObject(cfg)

    # container
    containers = {}
    container_1_cfg = RigidObjectCfg(
        prim_path=f"/World/Origin1/Container_1",
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
        prim_path=f"/World/Origin2/Container_2",
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
        prim_path=f"/World/Origin3/Container_3",
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
        prim_path=f"/World/Origin4/Container_4",
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
    for i, origin in enumerate(origins):
        index = i+5
        if index < len(origins):
            suitcase_cfg = RigidObjectCfg(
                prim_path=f"/World/Origin{index}/Suitcase",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=("/home/yu/IsaacLab/source/isaaclab_assets/manibot/suitcase.usd"),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        # disable_gravity=True,
                        # rigid_body_enabled=True,
                        # kinematic_enabled=True,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(random.random(), random.random(), random.random()), 
                        metallic=0.2
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(10, 10, 0.0)),
            )
            suitcase = RigidObject(cfg=suitcase_cfg)
            suitcases[f"suitcase_{i}"] = suitcase

    # Create a ray-caster sensor
    ray_caster_cfg = RayCasterCfg(
        prim_path="/World/Origin0/ball",
        offset=RayCasterCfg.OffsetCfg(pos=(box_size[1]/2, box_size[2]/2, box_size[0]+5)),
        mesh_prim_paths=["/World/Ground", "/World/Origin.*/Suitcase"],
        pattern_cfg=patterns.GridPatternCfg(resolution=1, size=(box_size[1]-1, box_size[2]-1)),
        attach_yaw_only=True,
        debug_vis=not args_cli.headless,
    )
    ray_caster = RayCaster(cfg=ray_caster_cfg)

    # return the scene information
    scene_entities = {"ball": ball, "suitcases": suitcases, "ray_caster": ray_caster, "box_size": box_size, "containers":containers}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    ray_caster: RayCaster = scene_entities["ray_caster"]
    ball: RigidObject = scene_entities["ball"]
    suitcases: dict = scene_entities["suitcases"]
    box_size = scene_entities["box_size"]
    containers = scene_entities["containers"]

    # Simulation step counter.
    dt = sim.get_physics_dt()
    count = 0

    # Get a list of all objects
    suitcases_list = list(suitcases.values())
    container_list = list(containers.values())

    # --- Packing Algorithm ---
    # Initialize packing problem with box size and items
    items = []
    suitcase_size = get_suitcase_size()
    for i in range(len(suitcases)):
        item = getSurfaceItem(suitcase_size[2], suitcase_size[0], suitcase_size[1])
        items.append(item)
    problem = PackingProblem(box_size, items)
    current_idx = 0  # The index of the object to be placed

    # Get the default state of the ball and randomize their positions (x,y).
    ball_default_state = ball.data.default_root_state.clone()

    while simulation_app.is_running():  
        # If there are still unplaced object, place the next one
        if current_idx < len(items) and count % 200 == 0:
            # print(f"hightmap",ray_caster.data.ray_hits_w[0, :, 2].cpu().numpy())
            raw_data = ray_caster.data.ray_hits_w[0, :, 2].cpu().numpy()
            raw_data[np.isinf(raw_data) | np.isnan(raw_data)] = 0
            problem.get_ray_caster_data(raw_data)
            transform = problem.autopack_oneitem(current_idx)
            transform_list = convert_transform_to_list(transform, device=args_cli.device)
            """
            When IsaacSim renders MeshCuboidCfg(size=(xSize,ySize,zSize)), 
            the model origin is usually at the center of the cube. The 
            raw_transform.position calculated by our algorithm is the 
            object frame origin position relative to the lower left 
            corner (0,0,0) of the container bottom surface (or world origin).
            """
            # get current item 
            item = items[current_idx]
            # Calculate the offset
            print("item.curr_geometry.x_size =", item.curr_geometry.x_size, 
                "item.curr_geometry.y_size =", item.curr_geometry.y_size,
                "item.curr_geometry.z_size =", item.curr_geometry.z_size)
            half_z = item.curr_geometry.z_size / 2.0   
            half_x = item.curr_geometry.x_size / 2.0   
            half_y = item.curr_geometry.y_size / 2.0 
            # add offset to transform_list
            transform_list[0] += half_x
            transform_list[1] += half_y
            transform_list[2] += half_z
            transform_tensor = torch.tensor(transform_list, device=args_cli.device)
            suitcases[f"suitcase_{current_idx}"].write_root_pose_to_sim(transform_tensor)  # apply sim data
            print(f"Item {current_idx} placed at transform: {transform_list}")
            current_idx += 1 

        # update buffers
        ball.write_root_pose_to_sim(ball_default_state[:, :7])
            
        # Step the simulation.
        sim.step()
        count += 1

        # Update the ray-caster.
        ray_caster.update(dt=dt, force_recompute=True)

        # update buffers
        for suitcase in suitcases_list:
            suitcase.update(dt)
        for container in container_list:
            container.update(dt)
        ball.update(dt)


def main():
    """Main function."""
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([70, 70, 70], [0.0, 0.0, 0.0])
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