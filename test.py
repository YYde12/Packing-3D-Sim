"""Launch Isaac Sim Simulator first."""
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Packing_3D.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import math
import random
import numpy as np
from pack import *
import time
import queue
import torch
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
import isaacsim.core.utils.prims as prim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

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

def design_scene():
    """Designs the scene with a container and 15 fixed objects."""
    # Ground-plane
    gp_cfg = sim_utils.GroundPlaneCfg()
    gp_cfg.func("/World/defaultGroundPlane", gp_cfg)

    # spawn distant light
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    # Container
    box_size = (15, 25, 25)
    '''
    container_cfg = RigidObjectCfg(
        prim_path="/World/Container",
        spawn=sim_utils.MeshCuboidCfg(
            size = (box_size[2], box_size[1], 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.7), metallic=0.1,opacity = 1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    container_object = RigidObject(cfg=container_cfg)
    '''

    # Create separate groups called "Origin1", "Origin2", "Origin3",...
    # Each group will have a robot in it
    origins = [[25, 25, 0.0], 
               [25, 35, 0.0], 
               [25, 45, 0.0], 
               [25, 55, 0.0],
               [25, 65, 0.0],
               [35, 25, 0.0],
               [35, 35, 0.0],
               [35, 45, 0.0],
               [35, 55, 0.0],
               [35, 65, 0.0],
               [45, 25, 0.0],
               [45, 35, 0.0],
               [45, 45, 0.0],
               [45, 55, 0.0],
               [45, 65, 0.0]
              ]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # 15 fixed objects
    item_objects = {}
    objects = [[8, 9, 9],
               [7, 6, 10],
               [8, 10, 9], 
               [9, 8, 5],
               [8, 5, 4],
               [7, 4, 4],
               [6, 6, 3],
               [6, 10, 10], 
               [9, 7, 6],
               [7, 6, 4],
               [7, 6, 4],
               [9, 6, 4], 
               [7, 7, 4],
               [10, 7, 4],
               [5, 5, 4]
              ]
    items = []
    for i, obj in enumerate(objects):
        # generate dimensions
        zSize = obj[0]
        xSize = obj[1]
        ySize = obj[2]
        # Generate object geometry with only the surface
        # print(obj)
        item = getSurfaceItem(zSize, xSize, ySize)
        items.append(item)
        # Rigid Object
        item_cfg = RigidObjectCfg(
            prim_path=f"/World/Origin{i}/item_{i}",
            spawn=sim_utils.MeshCuboidCfg(
                size=(xSize, ySize, zSize),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),   
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(random.random(), random.random(), random.random()), 
                    metallic=0.2
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )
        item_object = RigidObject(cfg=item_cfg)
        item_objects[f"item_{i}"] = item_object

    # Combine container and objects into a single dictionary for scene entities.
    scene_entities = {# "container": container_object,
                      "item_objects": item_objects,
                      "box_size": box_size,
                      "packing_items": items
                      }
    return scene_entities

def run_simulator(sim: SimulationContext, entities: dict):
    """Runs the simulation loop and executes the packing algorithm.
    PackingAlgorithm:
    - Create an instance of the PackingProblem class (using the size of the container and packing_items)
    - Call autopack_oneitem() in turn to calculate the placement transform of each object
    - Apply the calculated transform to the corresponding IsaacSim object
    """
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Extract objects for convenience.
    # container = entities["container"]
    item_objects = entities["item_objects"] 
    box_size = entities["box_size"]
    items = entities["packing_items"]

    # Get a list of all objects
    item_list = list(item_objects.values())

    # --- Packing Algorithm ---
    # Initialize packing problem with box size and items
    problem = PackingProblem(box_size, items)
    current_idx = 0  # The index of the object to be placed

    while simulation_app.is_running():
        # If there are still unplaced object, place the next one
        if current_idx < len(items):
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
            # print("item.curr_geometry.x_size =", item.curr_geometry.x_size, 
            #       "item.curr_geometry.y_size =", item.curr_geometry.y_size,
            #       "item.curr_geometry.z_size =", item.curr_geometry.z_size)
            # half_z = item.curr_geometry.z_size / 2.0   
            # half_x = item.curr_geometry.x_size / 2.0   
            # half_y = item.curr_geometry.y_size / 2.0 
            # add offset to transform_list
            # transform_list[0] += half_x
            # transform_list[1] += half_y
            # transform_list[2] += half_z
            # add offset to transform_list  
            centroid = item.curr_geometry.centroid()
            transform_list[0] += centroid.x
            transform_list[1] += centroid.y
            transform_list[2] += centroid.z
            # print("transform list", transform_list)
            transform_tensor = torch.tensor(transform_list, device=args_cli.device)
            # print(transform_tensor)
            item_objects[f"item_{current_idx}"].write_root_pose_to_sim(transform_tensor)  # apply sim data
            print(f"Item {current_idx} placed at transform: {transform_list}")
            current_idx += 1  
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt  
        count += 1
        # Update states
        # container.update(sim_dt)
        # update buffers
        for item in item_list:
            item.update(sim_dt)

        # print the root position
        '''
        if count % 50 == 0:
            for i, item in enumerate(item_list):
                pos = item.data.root_state_w[0, :3]
                print(f"Object {i} root position: {pos}")
        '''  
def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera 
    sim.set_camera_view(eye=[40, 40, 40], target=[0, 0, 0])
    # Build the scene with container and objects.
    scene_entities = design_scene()
    sim.reset()
    print("[INFO]: Setup complete...")
    # Run the simulation loop.
    run_simulator(sim, scene_entities)

if __name__ == "__main__":
    # Run the main simulation
    main()
    # close sim app
    simulation_app.close()
