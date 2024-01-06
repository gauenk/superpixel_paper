# -- misc --
import os,copy
dcopy = copy.deepcopy
import torch as th
from pytorch_lightning import Callback


class SaveCheckpointListByEpochs(Callback):

    def __init__(self,uuid,outdir,save_epochs):
        super().__init__()
        self.uuid = uuid
        self.outdir = outdir
        self.save_epochs = []
        self.save_interval = -1
        if "-" in save_epochs:
            self.save_epochs = [int(s) for s in save_epochs.split("-")]
            self.save_type = "list"
        elif save_epochs.startswith("by"):
            self.save_interval = int(save_epochs.split("by")[-1])
            self.save_type = "interval"

    def on_train_epoch_end(self, trainer, pl_module):
        uuid = self.uuid
        epoch = trainer.current_epoch
        if self.save_type == "list":
            if not(epoch in self.save_epochs): return
            path = Path(self.outdir / ("%s-save-epoch=%02d.ckpt" % (uuid,epoch)))
            trainer.save_checkpoint(str(path))
        elif self.save_type == "interval":
            if not((epoch+1) % self.save_interval == 0): return
            path = Path(self.outdir / ("%s-save-epoch=%02d.ckpt" % (uuid,epoch)))
            trainer.save_checkpoint(str(path))

class SaveCheckpointListBySteps(Callback):

    def __init__(self,uuid,outdir,save_steps,nkeep=-1):
        super().__init__()
        self.uuid = uuid
        self.outdir = outdir
        self.save_steps = []
        self.save_interval = -1
        self.nkeep = nkeep
        if "-" in save_steps:
            self.save_steps = [int(s) for s in save_steps.split("-")]
            self.save_type = "list"
            print("Saving Checkpoint via Steps [list]: [%s,%d]"
                  % (str(self.save_steps),nkeep))
        elif save_steps.startswith("by"):
            self.save_interval = int(save_steps.split("by")[-1])
            self.save_type = "interval"
            print("Saving Checkpoint via Steps [by]: [%d,%d]"
                  % (self.save_interval,nkeep))

    def on_train_batch_end(self, trainer, pl_module, *args):
        if not(rank_zero_only.rank == 0): return
        uuid = self.uuid
        step = trainer.global_step
        if step == 0: return
        if self.save_type == "list":
            if not(step in self.save_steps): return
            path = Path(self.outdir / ("%s-save-global_step=%02d.ckpt" % (uuid,step)))
            trainer.save_checkpoint(str(path))
        elif self.save_type == "interval":
            if not(step % self.save_interval == 0): return
            path = Path(self.outdir / ("%s-save-global_step=%02d.ckpt" % (uuid,step)))
            trainer.save_checkpoint(str(path))
        self.save_only_nkeep(step)

    def save_only_nkeep(self,step):
        if not(rank_zero_only.rank == 0): return
        if self.nkeep <= 0: return
        if self.save_type != "interval": return
        uuid = self.uuid
        nevents = step//self.save_interval+1
        start = max(1,nevents-self.nkeep-1)
        for i in range(start,nevents-self.nkeep):
            step_i = i*self.save_interval
            path = Path(self.outdir / ("%s-save-global_step=%02d.ckpt" % (uuid,step_i)))
            if path.exists(): os.remove(str(path.resolve()))

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def _accumulate_results(self,each_me):
        for key,val in each_me.items():
            # print(key,val)
            if not(key in self.metrics):
                self.metrics[key] = []
            if hasattr(val,"ndim"):
                ndim = val.ndim
                if isinstance(val,th.tensor):
                    val = val.cpu().numpy().item()
                else:
                    val = val
            self.metrics[key].append(val)

    def on_train_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_train_batch_end(self, trainer, pl_module, outs,
                           batch, batch_idx, dataloader_idx=0):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_validation_batch_end(self, trainer, pl_module, outs,
                                batch, batch_idx, dataloader_idx=0):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_batch_end(self, trainer, pl_module, outs,
                          batch, batch_idx, dataloader_idx=0):
        self._accumulate_results(outs)

