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

# def define_sensor() -> RayCaster:
#     """Defines the ray-caster sensor to add to the scene."""
#     # Create a ray-caster sensor
#     ray_caster_cfg = RayCasterCfg(
#         prim_path="/World/Origin0/ball",
#         offset=RayCasterCfg.OffsetCfg(pos=(1, 1, 26)),
#         mesh_prim_paths=["/World/ground", "/World/Origin.*/MovingCuboid"],
#         pattern_cfg=patterns.GridPatternCfg(resolution=1, size=(1.0, 1.0)),
#         attach_yaw_only=True,
#         debug_vis=not args_cli.headless,
#     )
#     ray_caster = RayCaster(cfg=ray_caster_cfg)

#     return ray_caster


def design_scene() -> dict:
    """Design the scene."""
    # -- Rough terrain
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd")
    cfg.func("/World/Ground", cfg)

    # spawn distant light
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    # Container
    box_size = (15, 25, 25)

    # Create separate groups called "Origin1", "Origin2", "Origin3",...
    # Each group will have a item in it
    origins = [[0.0, 0.0, 0.0], 
               [25, 35, 0.0], 
               [25, 45, 0.0], 
               [25, 55, 0.0],
               [25, 65, 0.0],
               [25, 75, 0.0]
               ]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # -- ball
    cfg = RigidObjectCfg(
        prim_path="/World/Origin0/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.005,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(random.random(), random.random(), random.random()), 
                    metallic=0.2
                ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    ball = RigidObject(cfg)

    # -- Moving Cuboid
    moving_cuboids = {}
    cuboids = [[8, 9, 9],
               [7, 6, 10],
               [8, 10, 9], 
               [9, 8, 5],
               [8, 5, 4],
               ]
    items = []
    for i, obj in enumerate(cuboids):
        # generate dimensions
        zSize = obj[0]
        xSize = obj[1]
        ySize = obj[2]
        # Generate object geometry with only the surface
        item = getSurfaceItem(zSize, xSize, ySize)
        items.append(item)
        index = i + 1
        moving_cuboid_cfg = RigidObjectCfg(
            prim_path=f"/World/Origin{index}/MovingCuboid",
            spawn=sim_utils.MeshCuboidCfg(
                size=(xSize, ySize, zSize),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(random.random(), random.random(), random.random()), 
                    metallic=0.2
                ),
            ),
        )
        moving_cuboid = RigidObject(moving_cuboid_cfg)
        moving_cuboids[f"moving_cuboid_{i}"] = moving_cuboid

    # Create a ray-caster sensor
    ray_caster_cfg = RayCasterCfg(
        prim_path="/World/Origin0/Ball",
        offset=RayCasterCfg.OffsetCfg(pos=(box_size[1]/2, box_size[2]/2, box_size[0])),
        mesh_prim_paths=["/World/Ground", "/World/Origin.*/MovingCuboid"],
        pattern_cfg=patterns.GridPatternCfg(resolution=1, size=(box_size[1]-1, box_size[2]-1)),
        attach_yaw_only=True,
        debug_vis=not args_cli.headless,
    )
    ray_caster = RayCaster(cfg=ray_caster_cfg)

    # return the scene information
    scene_entities = {"ball": ball, "moving_cuboids": moving_cuboids, "ray_caster": ray_caster, "box_size": box_size,
                      "packing_items": items}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    ray_caster: RayCaster = scene_entities["ray_caster"]
    ball: RigidObject = scene_entities["ball"]
    moving_cuboids: dict = scene_entities["moving_cuboids"]
    box_size = scene_entities["box_size"]
    items = scene_entities["packing_items"]

    dt = sim.get_physics_dt()

    # Simulation step counter.
    count = 0

    # --- Packing Algorithm ---
    # Initialize packing problem with box size and items
    problem = PackingProblem(box_size, items)
    current_idx = 0  # The index of the object to be placed

    # Get a list of all objects
    moving_cuboids_list = list(moving_cuboids.values())

    # Get the default state of the ball and randomize their positions (x,y).
    ball_default_state = ball.data.default_root_state.clone()
    

    # Compute the initial cuboid state based on the ball state.
    cuboid_default_state = ball_default_state.clone()
    cuboid_default_state[:, 0] = 40
    cuboid_default_state[:, 1] = 40
    cuboid_default_state[:, 2] = 0
    # Write the initial cuboid pose
    for moving_cuboid in moving_cuboids_list:
        moving_cuboid.write_root_pose_to_sim(cuboid_default_state[:, :7])


    while simulation_app.is_running():
        # If there are still unplaced object, place the next one
        if current_idx < len(items) and count % 100 == 0:
            # print("ray_caster.data.ray_hits_w:", ray_caster.data.ray_hits_w)
            problem.get_ray_caster_data(ray_caster.data.ray_hits_w[0, :, 2].cpu().numpy())
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
            moving_cuboids[f"moving_cuboid_{current_idx}"].write_root_pose_to_sim(transform_tensor)  # apply sim data
            print(f"Item {current_idx} placed at transform: {transform_list}")
            current_idx += 1  
            
        # update buffers
        ball.write_root_pose_to_sim(ball_default_state[:, :7])
        ball.update(dt)
            
        # Step the simulation.
        sim.step()
        count += 1

        # Update the ray-caster.
        ray_caster.update(dt=dt, force_recompute=True)

        # update buffers
        for moving_cuboid in moving_cuboids_list:
            moving_cuboid.update(dt)


def main():
    """Main function."""
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([50, 50, 40], [0.0, 0.0, 0.0])
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
