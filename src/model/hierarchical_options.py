"""Compatibility imports for the implemented hiltl-product-v2 option protocol.

The previous disconnected prototype is preserved in
``docs/archive/legacy-option-skeleton.zip``. Executable options live in
paper_options/PaperHiLTL and train_paper, with actual primitive and SMDP updates.
"""
from model.paper_options import available_options, distance_to, option_transition, smdp_targets

__all__ = ['available_options', 'distance_to', 'option_transition', 'smdp_targets']
