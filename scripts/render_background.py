"""
Author: Shuying Deng
Date: 2024-03-12 23:05:19
LastEditTime: 2024-03-21 14:03:12
Description: 

    All info based on Stack env. 
    OnlyRobot env is the version that directly deletes the block based on Stack env.

    Check the "robosuite/robosuite/environments/manipulation/only_robot_stack.py" before you run the code.
    `OnlyRobot` should be import in "robosuite/robosuite/__init__.py"

    how to run the code:
        cd /home/dsy/shared/demo_generate/robosuite/scripts        
        python onlyrobot_background.py --exp_name=test --env=OnlyRobot

        path to save the data: robosuite/exp/(exp_name)    
"""

import argparse

import imageio
import numpy as np
import os
import json

import open3d as o3d
from traitlets import Instance

import robosuite.macros as macros
from robosuite import make
import robosuite.utils.camera_utils as camera_utils


# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--env", type=str, default="OnlyRobot")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--camera", type=str, default="agentview", help="Name of camera to render"
    )
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--height", type=int, default=84)
    parser.add_argument("--width", type=int, default=84)
    parser.add_argument("--skip_frame", type=int, default=1)
    args = parser.parse_args()

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../backgrounds/" + str(args.exp_name),
    )
    os.makedirs(output_path, exist_ok=True)

    env = make(
        args.env,
        args.robots,
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=args.camera,
        camera_depths=True,
        camera_heights=args.height,
        camera_widths=args.width,
    )

    obs = env.reset()
    ndim = env.action_dim

    camera_intrinsic = camera_utils.get_camera_intrinsic_matrix(
        env.sim, args.camera, args.height, args.width
    )
    intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_matrix.set_intrinsics(
        args.width,
        args.height,
        camera_intrinsic[0, 0],
        camera_intrinsic[1, 1],
        camera_intrinsic[0, 2],
        camera_intrinsic[1, 2],
    )
    extrinsic_matrix = camera_utils.get_camera_extrinsic_matrix(env.sim, args.camera)

    rgb_writer = imageio.get_writer(output_path + "/rgb.mp4", fps=20)

    for i in range(args.timesteps):
        action = np.array([0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1])

        obs, reward, done, info = env.step(action)

        if i % args.skip_frame == 0:
            if i == args.timesteps - 1:
                rgb_frame_scene = obs[args.camera + "_image"]
                imageio.imwrite(
                    os.path.join(output_path + f"/background_{args.width}.png"),
                    rgb_frame_scene,
                )
                print("Saved image for last frame")

            print("frame #{}".format(i))

        if done:
            break

    env.close()
    rgb_writer.close()
