from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph_dataset_index.csv"
)

GRAPH_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graphs"
)


BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 0.001

HIDDEN_DIM = 64

NUM_EMBEDDINGS = 120

WEIGHT_DECAY = 1e-5

RANDOM_SEED = 42


torch.manual_seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)


if torch.backends.mps.is_available():

    DEVICE = torch.device(
        "mps"
    )

else:

    DEVICE = torch.device(
        "cpu"
    )


class CrystalGraphDataset(
    Dataset
):

    def __init__(
        self,
        index_df,
        graph_dir,
    ):

        super().__init__()

        self.index_df = (
            index_df
            .reset_index(
                drop=True
            )
        )

        self.graph_dir = graph_dir


    def len(
        self,
    ):

        return len(
            self.index_df
        )


    def get(
        self,
        idx,
    ):

        row = (
            self.index_df
            .iloc[idx]
        )

        graph_file = (
            self.graph_dir
            / row["graph_file"]
        )

        graph = torch.load(
            graph_file,
            weights_only=False,
        )


        target = (
            graph.y_voltage
            .clone()
            .float()
        )


        fields_to_remove = [
            "battery_id",
            "framework_formula",
            "num_atoms_original",
            "y_voltage",
            "y_capacity",
            "y_volume",
        ]


        for field in fields_to_remove:

            if field in graph:

                del graph[
                    field
                ]


        graph.y = target


        return graph


class VoltageGNN(
    nn.Module
):

    def __init__(
        self,
    ):

        super().__init__()


        self.element_embedding = (
            nn.Embedding(
                NUM_EMBEDDINGS,
                HIDDEN_DIM,
            )
        )


        self.extra_feature_layer = (
            nn.Sequential(
                nn.Linear(
                    2,
                    HIDDEN_DIM,
                ),
                nn.ReLU(),
            )
        )


        self.conv1 = GCNConv(
            HIDDEN_DIM,
            HIDDEN_DIM,
        )

        self.conv2 = GCNConv(
            HIDDEN_DIM,
            HIDDEN_DIM,
        )

        self.conv3 = GCNConv(
            HIDDEN_DIM,
            HIDDEN_DIM,
        )


        self.regressor = (
            nn.Sequential(

                nn.Linear(
                    HIDDEN_DIM,
                    HIDDEN_DIM,
                ),

                nn.ReLU(),

                nn.Dropout(
                    0.15
                ),

                nn.Linear(
                    HIDDEN_DIM,
                    32,
                ),

                nn.ReLU(),

                nn.Linear(
                    32,
                    1,
                ),
            )
        )


    def forward(
        self,
        data,
    ):

        z = data.z.long()

        x_extra = (
            data.x.float()
        )


        element_features = (
            self.element_embedding(
                z
            )
        )


        extra_features = (
            self.extra_feature_layer(
                x_extra
            )
        )


        x = (
            element_features
            +
            extra_features
        )


        x = self.conv1(
            x,
            data.edge_index,
        )

        x = torch.relu(
            x
        )


        x = self.conv2(
            x,
            data.edge_index,
        )

        x = torch.relu(
            x
        )


        x = self.conv3(
            x,
            data.edge_index,
        )

        x = torch.relu(
            x
        )


        graph_embedding = (
            global_mean_pool(
                x,
                data.batch,
            )
        )


        prediction = (
            self.regressor(
                graph_embedding
            )
        )


        prediction = (
            prediction
            .view(-1)
        )


        return prediction


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_function,
):

    model.train()

    total_loss = 0.0

    total_samples = 0


    for batch in loader:

        batch = batch.to(
            DEVICE
        )


        optimizer.zero_grad()


        prediction = model(
            batch
        )


        target = (
            batch.y
            .view(-1)
            .float()
        )


        loss = (
            loss_function(
                prediction,
                target,
            )
        )


        loss.backward()


        optimizer.step()


        current_batch_size = (
            target.size(
                0
            )
        )


        total_loss += (
            loss.item()
            * current_batch_size
        )

        total_samples += (
            current_batch_size
        )


    mean_loss = (
        total_loss
        / total_samples
    )


    return mean_loss


def evaluate(
    model,
    loader,
):

    model.eval()


    prediction_list = []

    target_list = []


    with torch.no_grad():

        for batch in loader:

            batch = batch.to(
                DEVICE
            )


            prediction = model(
                batch
            )


            target = (
                batch.y
                .view(-1)
                .float()
            )


            prediction_list.extend(
                prediction
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )


            target_list.extend(
                target
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )


    predictions = np.array(
        prediction_list,
        dtype=float,
    )

    targets = np.array(
        target_list,
        dtype=float,
    )


    r2 = r2_score(
        targets,
        predictions,
    )


    mae = mean_absolute_error(
        targets,
        predictions,
    )


    rmse = np.sqrt(
        mean_squared_error(
            targets,
            predictions,
        )
    )


    return (
        r2,
        mae,
        rmse,
    )


def inspect_first_graph(
    dataset,
):

    graph = dataset[
        0
    ]


    print()
    print(
        "First graph check:"
    )


    print(
        "Fields:",
        graph.keys(),
    )


    print(
        "Node feature shape:",
        graph.x.shape,
    )


    print(
        "Atomic number shape:",
        graph.z.shape,
    )


    print(
        "Edge index shape:",
        graph.edge_index.shape,
    )


    print(
        "Edge attr shape:",
        graph.edge_attr.shape,
    )


    print(
        "Target:",
        graph.y,
    )


def main():

    print(
        "Device:",
        DEVICE,
    )


    index_df = pd.read_csv(
        INDEX_FILE
    )


    valid_status = [
        "success",
        "existing",
    ]


    index_df = (
        index_df[
            index_df[
                "status"
            ].isin(
                valid_status
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    print(
        "Valid graphs:",
        len(
            index_df
        ),
    )


    print(
        "Unique frameworks:",
        index_df[
            "framework_formula"
        ].nunique(),
    )


    groups = (
        index_df[
            "framework_formula"
        ]
        .values
    )


    group_cv = GroupKFold(
        n_splits=5
    )


    splits = list(
        group_cv.split(
            index_df,
            groups=groups,
        )
    )


    train_index, test_index = (
        splits[
            0
        ]
    )


    train_df = (
        index_df
        .iloc[
            train_index
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    test_df = (
        index_df
        .iloc[
            test_index
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    print()
    print(
        "Fold: 1"
    )


    print(
        "Train samples:",
        len(
            train_df
        ),
    )


    print(
        "Test samples:",
        len(
            test_df
        ),
    )


    print(
        "Train frameworks:",
        train_df[
            "framework_formula"
        ].nunique(),
    )


    print(
        "Test frameworks:",
        test_df[
            "framework_formula"
        ].nunique(),
    )


    overlap = set(
        train_df[
            "framework_formula"
        ]
    ).intersection(
        set(
            test_df[
                "framework_formula"
            ]
        )
    )


    print(
        "Framework overlap:",
        len(
            overlap
        ),
    )


    train_dataset = (
        CrystalGraphDataset(
            train_df,
            GRAPH_DIR,
        )
    )


    test_dataset = (
        CrystalGraphDataset(
            test_df,
            GRAPH_DIR,
        )
    )


    inspect_first_graph(
        train_dataset
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


    model = VoltageGNN().to(
        DEVICE
    )


    print()
    print(
        "Model:"
    )

    print(
        model
    )


    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    print()
    print(
        "Trainable parameters:",
        total_parameters,
    )


    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
    )


    loss_function = (
        nn.MSELoss()
    )


    print()
    print(
        "Training started"
    )


    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_function,
            )
        )


        if (
            epoch == 1
            or
            epoch % 5 == 0
        ):

            (
                test_r2,
                test_mae,
                test_rmse,
            ) = evaluate(
                model,
                test_loader,
            )


            print(
                f"Epoch {epoch:03d} "
                f"TrainLoss={train_loss:.4f} "
                f"TestR2={test_r2:.4f} "
                f"TestMAE={test_mae:.4f} "
                f"TestRMSE={test_rmse:.4f}"
            )


    (
        final_r2,
        final_mae,
        final_rmse,
    ) = evaluate(
        model,
        test_loader,
    )


    print()
    print(
        "Final Fold 1 results:"
    )


    print(
        "R2:",
        round(
            final_r2,
            4,
        )
    )


    print(
        "MAE:",
        round(
            final_mae,
            4,
        ),
        "V",
    )


    print(
        "RMSE:",
        round(
            final_rmse,
            4,
        ),
        "V",
    )


    print()
    print(
        "XGBoost Framework Group CV benchmark:"
    )

    print(
        "R2: 0.4935"
    )

    print(
        "MAE: 0.4185 V"
    )


    print()
    print(
        "STEP 13 completed."
    )


if __name__ == "__main__":

    main()