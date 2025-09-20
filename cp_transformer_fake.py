from cp_transformer_fine_tune import RoformerFineTune
from cp_transformer import RoFormerSymbolicTransformer


class FakeModel(RoformerFineTune):
    def __init__(self, task_name):
        super().__init__(compress_ratio_l=1, compress_ratio_r=1, lr=None)
        self.save_name = 'gt_' + task_name
        self.wrapped_model = RoFormerSymbolicTransformer(
            size=0,
            with_velocity=False,
        )
