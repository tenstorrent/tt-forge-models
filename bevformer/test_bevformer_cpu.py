# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from third_party.tt_forge_models.bevformer.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)
from tests.jax.single_chip.models.bevformer.model_utils.model_utils import (
    BEV_wrapper_base,
    build_from_input_image_base,
    BEV_wrapper_v2,
    build_from_input_image_v2,
)


def test_bevformer():

    # Load model and inputs
    variant = ModelVariant.BEVFORMER_V2_R50_T1_BASE
    loader = ModelLoader(variant)
    framework_model = loader.load_model()
    inputs_dict = loader.load_inputs()

    if variant in [
        ModelVariant.BEVFORMER_V2_R50_T1_BASE,
        ModelVariant.BEVFORMER_V2_R50_T1,
        ModelVariant.BEVFORMER_V2_R50_T2,
        ModelVariant.BEVFORMER_V2_R50_T8,
    ]:
        built = build_from_input_image_v2(inputs_dict)
        wrapper_bev_model = BEV_wrapper_v2(
            framework_model,
            built["filename"],
            built["box_mode_3d"],
            built["box_type_3d"],
            built["img_norm_cfg"],
            built["pts_filename"],
            built["lidar2img"],
            built["ori_shapes"],
            built["lidar2cam"],
            built["img_shapes"],
            built["pad_shape"],
        )
        wrapper_bev_model.eval()
        result_wrapper = wrapper_bev_model(
            built["ori_shapes_tensor"],
            built["img_shapes_tensor"],
            built["lidar2img_stacked_tensor"],
            built["lidar2cam_stacked_tensor"],
            built["pad_shapes_tensor"],
            built["img_pybuda"],
            built["ego2global_translation"],
            built["ego2global_rotation"],
            built["lidar2ego_translation"],
            built["lidar2ego_rotation"],
            built["timestamp"],
        )
        print("result_wrapper = ", result_wrapper)
    else:
        built = build_from_input_image_base(inputs_dict)
        wrapper_bev_model = BEV_wrapper_base(
            framework_model,
            built["filename"],
            built["box_mode_3d"],
            built["box_type_3d"],
            built["img_norm_cfg"],
            built["pts_filename"],
            built["can_bus"],
            built["lidar2img"],
            built["ori_shapes"],
            built["lidar2cam"],
            built["img_shapes"],
            built["pad_shape"],
        )
        wrapper_bev_model.eval()
        result_wrapper = wrapper_bev_model(
            built["ori_shapes_tensor"],
            built["img_shapes_tensor"],
            built["lidar2img_stacked_tensor"],
            built["lidar2cam_stacked_tensor"],
            built["pad_shapes_tensor"],
            built["img_pybuda"],
        )
        print("result_wrapper = ", result_wrapper)


if __name__ == "__main__":
    test_bevformer()
