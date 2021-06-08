import os

import torch
import numpy as np
from pytorch_lightning.callbacks.base import Callback


class SaveNodeEncodings(Callback):
    def __init__(self):
        super().__init__()
        self.dirpath = None
        self.savepath = None

    def on_test_start(self, trainer, pl_module):
        save_dir = (getattr(trainer, 'weights_save_path', None)
                    or getattr(trainer.logger, 'save_dir', None)
                    or trainer.default_root_dir)

        version = trainer.logger.version if isinstance(
            trainer.logger.version, str) else f'version_{trainer.logger.version}'
        ckpt_path = os.path.join(
            save_dir,
            trainer.logger.name,
            version,
            "node_encodings"
        )
        self.dirpath = ckpt_path
        self.savepath = os.path.join(self.dirpath, 'node_encodings.npz')

    def on_test_end(self, trainer, pl_module):
        test_node_encodings = getattr(pl_module, 'test_encodings', None)
        if test_node_encodings is not None:
            print('Saving node encodings to {}...'.format(self.savepath))
            if not os.path.exists(self.dirpath):
                os.makedirs(self.dirpath, exist_ok=True)
            np.savez(self.savepath, **test_node_encodings)
        else:
            print('No node encoding available!')