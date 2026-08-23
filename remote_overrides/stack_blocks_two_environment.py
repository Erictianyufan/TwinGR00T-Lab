import argparse
from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase

class StackBlocksTwoEnvironment(ExampleEnvironmentBase):
    name = "stack_blocks_two"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sim.spawners.from_files import UsdFileCfg
        from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, CollisionPropertiesCfg
        from isaaclab.sim.spawners.materials import VisualMaterialCfg
        import isaaclab.sim.spawners as sp
        from isaaclab.sensors.camera import TiledCameraCfg
        import isaaclab.sim as sim_utils
        
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.assets.asset import Asset

        from manip_eval_tasks.embodiments.aloha_agilex.aloha_agilex import AlohaAgilexEmbodiment, ReplayAlohaActionsCfg
        from manip_eval_tasks.tasks.stack_task import StackMultiObjectTask
        from manip_eval_tasks.assets.object_library import ProceduralTable, ProceduralBlock

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(args_cli.enable_cameras)
        if args_cli.embodiment in ["aloha"]:
            embodiment.action_config = ReplayAlohaActionsCfg()
            embodiment.set_initial_pose(
                Pose(position_xyz=(0.0, -0.65, 0.0), rotation_wxyz=(0.707, 0.0, 0.0, 0.707))
            )
        else:
            raise ValueError(f"Unsupported embodiment type: {args_cli.embodiment}")

        block1_asset = self.asset_registry.get_asset_by_name("robotwin_block")(prim_path="{ENV_REGEX_NS}/block1", color=(1.0, 0.0, 0.0), size=(0.05, 0.05, 0.05), name='block1')
        block2_asset = self.asset_registry.get_asset_by_name("robotwin_block")(prim_path="{ENV_REGEX_NS}/block2", color=(0.0, 1.0, 0.0), size=(0.05, 0.05, 0.05), name='block2')

        for block_asset in (block1_asset, block2_asset):
            block_asset.object_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.25,
                angular_damping=0.8,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            )
            block_asset.object_cfg.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
                contact_offset=0.003,
                rest_offset=0.0,
            )
            block_asset.object_cfg.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
                friction_combine_mode="max",
                restitution_combine_mode="min",
            )

        table = self.asset_registry.get_asset_by_name("robotwin_table")()
        scene = Scene(assets=[table, block1_asset, block2_asset])        

        task = StackMultiObjectTask(
            stack_order_list=[block1_asset, block2_asset],
            episode_length_s=25.0,
            stack_offset=0.05,
            exp_xy=0.015,
            eps_z=0.010,
        )

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=None,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--enable_cameras", type=bool, default=False)
        parser.add_argument("--embodiment", type=str, default="aloha")
