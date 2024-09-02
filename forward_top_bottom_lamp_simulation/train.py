import time
import matplotlib.pyplot as plt

import hydra
from hydra.utils import to_absolute_path

from dgl.dataloading import GraphDataLoader

from omegaconf import DictConfig

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

import wandb

from thermoforming_dataset import ThermoformingDataset #MYDATSET
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
    initialize_wandb,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.models.meshgraphnet import MeshGraphNet
import numpy as np


class MGNTrainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.rank_zero_logger = rank_zero_logger

        self.amp = cfg.amp
        # MGN with recompute_activation currently supports only SiLU activation function.
        mlp_act = "relu"
        if cfg.recompute_activation:
            rank_zero_logger.info(
                "Setting MLP activation to SiLU required by recompute_activation."
            )
            mlp_act = "silu"

        # instantiate dataset
        rank_zero_logger.info("Loading the training dataset...")
        self.dataset = ThermoformingDataset(
            name="termoforming_train",
            data_dir=to_absolute_path(cfg.data_dir),
            split="train",
            num_samples=cfg.num_training_samples,
            num_workers=cfg.num_dataset_workers,
        )

        # instantiate validation dataset
        rank_zero_logger.info("Loading the validation dataset...")
        self.validation_dataset = ThermoformingDataset(
            name="termoforming_validation",
            data_dir=to_absolute_path(cfg.data_dir),
            split="validation",
            num_samples=cfg.num_validation_samples,
            num_workers=cfg.num_dataset_workers,
        )

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=self.dist.world_size > 1,
            num_workers=cfg.num_dataloader_workers,
        )

        # instantiate validation dataloader
        self.validation_dataloader = GraphDataLoader(
            self.validation_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=False,
            num_workers=cfg.num_dataloader_workers,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            cfg.input_dim_nodes,
            cfg.input_dim_edges,
            cfg.output_dim,
            aggregation=cfg.aggregation,
            hidden_dim_node_encoder=cfg.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=cfg.hidden_dim_edge_encoder,
            hidden_dim_node_decoder=cfg.hidden_dim_node_decoder,
            mlp_activation_fn=mlp_act,
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
        )
        if cfg.jit:
            if not self.model.meta.jit:
                raise ValueError("MeshGraphNet is not yet JIT-compatible.")
            self.model = torch.jit.script(self.model).to(self.dist.device)
        else:
            self.model = self.model.to(self.dist.device)

        # distributed data parallel for multi-node training
        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        # enable train mode
        self.model.train()

        # instantiate optimizer, and scheduler
        self.optimizer = None
        try:
            if cfg.use_apex:
                from apex.optimizers import FusedAdam

                self.optimizer = FusedAdam(self.model.parameters(), lr=cfg.lr)
        except ImportError:
            rank_zero_logger.warning(
                "NVIDIA Apex (https://github.com/nvidia/apex) is not installed, "
                "FusedAdam optimizer will not be used."
            )
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        rank_zero_logger.info(f"Using {self.optimizer.__class__.__name__} optimizer")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        # load checkpoint
        if self.dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )

    def train(self, graph):
        self.optimizer.zero_grad()
        loss = self.forward(graph)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def forward(self, graph):
        # forward pass
        with autocast(enabled=self.amp):
            if torch.isnan(graph.ndata["x"]).any() or torch.isinf(graph.ndata["x"]).any():
                print("NaN or Inf found in graph.ndata[\"x\"] tensor")
            if torch.isnan(graph.edata["x"]).any() or torch.isinf(graph.edata["x"]).any():
                print("NaN or Inf found in graph.edata[\"x\"] tensor")
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)

            C_pred = pred[:, [0]]
            C_gt = graph.ndata["y"][:, [0]]
            maxC_pred = pred[:, [1]]
            maxC_gt = graph.ndata["y"][:, [1]]
            maxC_mean = torch.mean(maxC_pred, dim=0)
            
            diff_norm = torch.norm(
                torch.flatten(C_pred) - torch.flatten(C_gt), p=2
            )
            y_norm = torch.norm(torch.flatten(C_gt), p=2)
            loss_C = diff_norm / y_norm

            loss_maxC = torch.sum((maxC_mean - maxC_gt) ** 2)

            return loss_C + loss_maxC#1.5 * loss_C + loss_maxC

    def backward(self, loss):
        # backward pass
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        lr = self.get_lr()
        wandb.log({"lr": lr})

    def get_lr(self):
        # get the learning rate
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    @torch.no_grad()
    def validation(self):
        error = 0
        for i, (graph, sid) in enumerate(self.validation_dataloader):
            graph = graph.to(self.dist.device)
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)

            C_pred = pred[:, [0]]
            C_gt = graph.ndata["y"][:, [0]]
            maxC_pred = pred[:, [1]]
            maxC_gt = graph.ndata["y"][:, [1]]
            maxC_mean = torch.mean(maxC_pred, dim=0)
            
            diff_norm = torch.norm(
                torch.flatten(C_pred) - torch.flatten(C_gt), p=2
            )
            y_norm = torch.norm(torch.flatten(C_gt), p=2)
            loss_C = diff_norm / y_norm

            loss_maxC = torch.sum((maxC_mean - maxC_gt) ** 2)

            error += (loss_C + loss_maxC) 
            
        error = error / len(self.validation_dataloader)
        wandb.log({"val_error (%)": error})
        self.rank_zero_logger.info(f"Validation loss: {error:10.3e}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # initialize loggers
    initialize_wandb(
        project="Thermoforming",
        entity="Modulus",
        name="Thermo-Training",
        #group="Thermo-DDP-Group",
        mode=cfg.wandb_mode,
    )  # Wandb logger

    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    rank_zero_logger.file_logging()

    trainer = MGNTrainer(cfg, rank_zero_logger)
    start = time.time()
    rank_zero_logger.info("Training started...")
    
    training_losses = []
    validation_losses = []

    for epoch in range(trainer.epoch_init, cfg.epochs):
        loss_agg = 0
        for i, (graph, sid) in enumerate(trainer.dataloader):
            graph = graph.to(dist.device)
            loss = trainer.train(graph)
            loss_agg += loss.detach().cpu().numpy()
        loss_agg /= len(trainer.dataloader)
        rank_zero_logger.info(
            f"epoch: {epoch}, loss: {loss_agg:10.3e}, lr: {trainer.get_lr()}, "
            f"time per epoch: {(time.time()-start):10.3e}"
        )
        wandb.log({"loss": loss_agg})
        # validation
        if dist.rank == 0:
            validation_loss = trainer.validation()
            validation_losses.append(validation_loss)
            
        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0 and (epoch + 1) % cfg.checkpoint_save_freq == 0:
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            rank_zero_logger.info(f"Saved model on rank {dist.rank}")
        start = time.time()
    rank_zero_logger.info("Training completed!")
    
    plt.plot(training_losses, label='Training loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()