from dgl import DGLGraph
from dgl.dataloading import GraphDataLoader

import hydra
from hydra.utils import to_absolute_path

import numpy as np
import pyvista as pv
import torch

from omegaconf import DictConfig

from thermoforming_dataset import ThermoformingDataset
from modulus.launch.logging import PythonLogger
from modulus.launch.utils import load_checkpoint
from modulus.models.meshgraphnet import MeshGraphNet

from utils import relative_lp_error

POWER_VALUES = ["Power_top", "Power_bottom"]

def dgl_to_pyvista(graph: DGLGraph):
    """
    Converts a DGL graph to a PyVista graph.

    Parameters:
    -----------
    graph: DGLGraph
        The input DGL graph.

    Returns:
    --------
    pv_graph:
        The output PyVista graph.
    """

    # Convert the DGL graph to a NetworkX graph
    nx_graph = graph.to_networkx(
        node_attrs=["pos", "C_pred", "C"]
    ).to_undirected()

    # Initialize empty lists for storing data
    points = []
    lines = []
    C_pred = []
    C = []

    # Iterate over the nodes in the NetworkX graph
    for node, attributes in nx_graph.nodes(data=True):
        # Append the node and attribute data to the respective lists
        points.append(attributes["pos"].numpy())
        C_pred.append(attributes["C_pred"].numpy())
        C.append(attributes["C"].numpy())

    # Add edges to the lines list
    for edge in nx_graph.edges():
        lines.extend([2, edge[0], edge[1]])

    # Initialize a PyVista graph
    pv_graph = pv.PolyData()

    # Assign the points, lines, and attributes to the PyVista graph
    pv_graph.points = np.array(points)
    pv_graph.lines = np.array(lines)
    pv_graph.point_data["C_pred"] = np.array(C_pred)
    pv_graph.point_data["C"] = np.array(C)

    return pv_graph

def prediction_to_str(predictions, ground_truth):
    """
    Converts a DGL graph to a string.

    Parameters:
    -----------
    graph: DGLGraph
        The input DGL graph.

    Returns:
    --------
    str_graph:
        The predicted output for each point position.
    """

    
    str_graph = ""

    str_graph += f"maxC prediction: {predictions[0]}, truth value: {ground_truth[0, 0]}\n"
   
    return str_graph


class ThermoformingRollout:
    """MGN inference on Thermoforming dataset"""

    def __init__(self, cfg: DictConfig, logger: PythonLogger):
        # set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        logger.info(f"Using {self.device} device")

        # instantiate dataset
        self.dataset = ThermoformingDataset(
            name="termoforming_test",
            data_dir=to_absolute_path(cfg.data_dir),
            split="test",
            num_samples=cfg.num_test_samples,
        )

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
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
            mlp_activation_fn="silu" if cfg.recompute_activation else "relu",
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
        )
        self.model = self.model.to(self.device)

        # enable train mode
        self.model.eval()

        # load checkpoint
        load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            device=self.device,
        )

    def predict(self, save_results=False):
        """
        Run the prediction process.

        Parameters:
        -----------
        save_results: bool
            Whether to save the results in form of a .vtp file, by default False


        Returns:
        --------
        None
        """
        self.pred, self.exact, self.faces, self.graphs = [], [], [], []

        error = 0
        for i, (graph, sid) in enumerate(self.dataloader):
            graph = graph.to(self.device)
            sid = sid.item()
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph).detach()
            self.logger.info(f"Processing sample ID {sid}")
            pred, gt = self.dataset.denormalize(
                pred, graph.ndata["y"], self.device
            )
            graph.ndata["C_pred"] = pred[:, 0]
            graph.ndata["C"] = gt[:, 0]
            graph.ndata["maxC_pred"] = pred[:, 1:]
            maxC_predictions = pred[:, 1:]            
            graph.ndata["maxC"] = gt[:, 1:]

            C_pred = pred[:, [0]]
            C_gt = gt[:, [0]]

            error_C = relative_lp_error(C_pred, C_gt)

            maxC_pred = pred[:, [1]]
            maxC_gt = gt[:, [1]]
            maxC_mean = torch.mean(maxC_pred, dim=0)

            error_maxC = torch.sum((maxC_mean - maxC_gt) ** 2)

            error = (error_C + error_maxC)
            self.logger.info(f"Test l2 error C: {error_C} l2 error node {error_C/len(C_pred)}")
            self.logger.info(f"Test squared error maxC: {error_maxC}")

            if save_results:
                pv_graph = dgl_to_pyvista(graph.cpu())
                pv_graph.save(f"graph_{sid}.vtp")
                str_graph = prediction_to_str(maxC_mean, gt[:, 1:])
                with open(f"graph_{sid}.txt", 'w') as f:
                    f.write(str_graph)




@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("main")  # General python logger
    logger.file_logging()

    logger.info("Rollout started...")
    rollout = ThermoformingRollout(cfg, logger)
    rollout.predict(save_results=True)


if __name__ == "__main__":
    main()