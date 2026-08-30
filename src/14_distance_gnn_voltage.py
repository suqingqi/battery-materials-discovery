from pathlib import Path

import copy

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
from torch_geometric.nn import global_mean_pool


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

MAX_EPOCHS = 100

PATIENCE = 15

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-5

HIDDEN_DIM = 64

EDGE_DIM = 32

NUM_EMBEDDINGS = 120

CUTOFF = 5.0

NUM_GAUSSIANS = 32

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


        remove_fields = [
            "battery_id",
            "framework_formula",
            "num_atoms_original",
            "y_voltage",
            "y_capacity",
            "y_volume",
        ]


        for field in remove_fields:

            if field in graph:

                del graph[
                    field
                ]


        graph.y = target


        return graph


class GaussianDistance(
    nn.Module
):

    def __init__(
        self,
        cutoff,
        num_gaussians,
    ):

        super().__init__()


        centers = torch.linspace(
            0.0,
            cutoff,
            num_gaussians,
        )


        self.register_buffer(
            "centers",
            centers,
        )


        spacing = (
            centers[1]
            -
            centers[0]
        )


        self.gamma = (
            1.0
            /
            float(
                spacing ** 2
            )
        )


    def forward(
        self,
        distance,
    ):

        distance = (
            distance
            .view(
                -1,
                1,
            )
        )


        gaussian = torch.exp(
            -self.gamma
            * (
                distance
                -
                self.centers
            ) ** 2
        )


        return gaussian


class CrystalMessageLayer(
    nn.Module
):

    def __init__(
        self,
        hidden_dim,
        edge_dim,
    ):

        super().__init__()


        self.message_network = (
            nn.Sequential(

                nn.Linear(
                    hidden_dim * 2
                    + edge_dim,
                    hidden_dim,
                ),

                nn.ReLU(),

                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),
            )
        )


        self.update_network = (
            nn.Sequential(

                nn.Linear(
                    hidden_dim * 2,
                    hidden_dim,
                ),

                nn.ReLU(),

                nn.Linear(
                    hidden_dim,
                    hidden_dim,
                ),
            )
        )


        self.norm = nn.LayerNorm(
            hidden_dim
        )


    def forward(
        self,
        x,
        edge_index,
        edge_features,
    ):

        source = (
            edge_index[
                0
            ]
        )

        target = (
            edge_index[
                1
            ]
        )


        source_features = (
            x[
                source
            ]
        )

        target_features = (
            x[
                target
            ]
        )


        message_input = torch.cat(
            [
                source_features,
                target_features,
                edge_features,
            ],
            dim=1,
        )


        messages = (
            self.message_network(
                message_input
            )
        )


        aggregated = torch.zeros(
            x.size(
                0
            ),
            x.size(
                1
            ),
            device=x.device,
            dtype=x.dtype,
        )


        aggregated.index_add_(
            0,
            target,
            messages,
        )


        degree = torch.zeros(
            x.size(
                0
            ),
            device=x.device,
            dtype=x.dtype,
        )


        degree.index_add_(
            0,
            target,
            torch.ones(
                target.size(
                    0
                ),
                device=x.device,
                dtype=x.dtype,
            ),
        )


        degree = (
            degree
            .clamp(
                min=1.0
            )
            .view(
                -1,
                1,
            )
        )


        aggregated = (
            aggregated
            /
            degree
        )


        update_input = torch.cat(
            [
                x,
                aggregated,
            ],
            dim=1,
        )


        update = (
            self.update_network(
                update_input
            )
        )


        x = (
            x
            +
            update
        )


        x = self.norm(
            x
        )


        return x


class DistanceAwareGNN(
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


        self.gaussian_distance = (
            GaussianDistance(
                CUTOFF,
                NUM_GAUSSIANS,
            )
        )


        self.edge_network = (
            nn.Sequential(

                nn.Linear(
                    NUM_GAUSSIANS,
                    EDGE_DIM,
                ),

                nn.ReLU(),

                nn.Linear(
                    EDGE_DIM,
                    EDGE_DIM,
                ),
            )
        )


        self.message1 = (
            CrystalMessageLayer(
                HIDDEN_DIM,
                EDGE_DIM,
            )
        )

        self.message2 = (
            CrystalMessageLayer(
                HIDDEN_DIM,
                EDGE_DIM,
            )
        )

        self.message3 = (
            CrystalMessageLayer(
                HIDDEN_DIM,
                EDGE_DIM,
            )
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

        z = (
            data.z
            .long()
        )


        extra = (
            data.x
            .float()
        )


        element_features = (
            self.element_embedding(
                z
            )
        )


        extra_features = (
            self.extra_feature_layer(
                extra
            )
        )


        x = (
            element_features
            +
            extra_features
        )


        distances = (
            data.edge_attr
            .view(
                -1
            )
            .float()
        )


        distance_basis = (
            self.gaussian_distance(
                distances
            )
        )


        edge_features = (
            self.edge_network(
                distance_basis
            )
        )


        x = self.message1(
            x,
            data.edge_index,
            edge_features,
        )


        x = self.message2(
            x,
            data.edge_index,
            edge_features,
        )


        x = self.message3(
            x,
            data.edge_index,
            edge_features,
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


        return (
            prediction
            .view(-1)
        )


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


        loss = loss_function(
            prediction,
            target,
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )


        optimizer.step()


        batch_size = (
            target.size(
                0
            )
        )


        total_loss += (
            loss.item()
            *
            batch_size
        )


        total_samples += (
            batch_size
        )


    return (
        total_loss
        /
        total_samples
    )


def evaluate(
    model,
    loader,
):

    model.eval()


    predictions = []

    targets = []


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


            predictions.extend(
                prediction
                .cpu()
                .numpy()
                .tolist()
            )


            targets.extend(
                target
                .cpu()
                .numpy()
                .tolist()
            )


    predictions = np.array(
        predictions
    )

    targets = np.array(
        targets
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


def make_train_validation_split(
    train_df,
):

    groups = (
        train_df[
            "framework_formula"
        ]
        .values
    )


    splitter = GroupKFold(
        n_splits=5
    )


    inner_splits = list(
        splitter.split(
            train_df,
            groups=groups,
        )
    )


    inner_train_idx, val_idx = (
        inner_splits[
            0
        ]
    )


    inner_train_df = (
        train_df
        .iloc[
            inner_train_idx
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    val_df = (
        train_df
        .iloc[
            val_idx
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    return (
        inner_train_df,
        val_df,
    )


def main():

    print(
        "Device:",
        DEVICE,
    )


    index_df = pd.read_csv(
        INDEX_FILE
    )


    index_df = (
        index_df[
            index_df[
                "status"
            ].isin(
                [
                    "success",
                    "existing",
                ]
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


    outer_cv = GroupKFold(
        n_splits=5
    )


    outer_splits = list(
        outer_cv.split(
            index_df,
            groups=groups,
        )
    )


    train_index, test_index = (
        outer_splits[
            0
        ]
    )


    outer_train_df = (
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


    train_df, val_df = (
        make_train_validation_split(
            outer_train_df
        )
    )


    print()
    print(
        "Outer Fold: 1"
    )


    print(
        "Train samples:",
        len(
            train_df
        ),
    )


    print(
        "Validation samples:",
        len(
            val_df
        ),
    )


    print(
        "Test samples:",
        len(
            test_df
        ),
    )


    train_frameworks = set(
        train_df[
            "framework_formula"
        ]
    )


    val_frameworks = set(
        val_df[
            "framework_formula"
        ]
    )


    test_frameworks = set(
        test_df[
            "framework_formula"
        ]
    )


    print(
        "Train-Val overlap:",
        len(
            train_frameworks
            &
            val_frameworks
        ),
    )


    print(
        "Train-Test overlap:",
        len(
            train_frameworks
            &
            test_frameworks
        ),
    )


    print(
        "Val-Test overlap:",
        len(
            val_frameworks
            &
            test_frameworks
        ),
    )


    train_dataset = (
        CrystalGraphDataset(
            train_df,
            GRAPH_DIR,
        )
    )


    val_dataset = (
        CrystalGraphDataset(
            val_df,
            GRAPH_DIR,
        )
    )


    test_dataset = (
        CrystalGraphDataset(
            test_df,
            GRAPH_DIR,
        )
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


    model = (
        DistanceAwareGNN()
        .to(
            DEVICE
        )
    )


    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
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


    best_val_mae = float(
        "inf"
    )

    best_epoch = 0

    best_state = None

    patience_counter = 0


    print()
    print(
        "Training started"
    )


    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        train_loss = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_function,
            )
        )


        (
            val_r2,
            val_mae,
            val_rmse,
        ) = evaluate(
            model,
            val_loader,
        )


        if (
            epoch == 1
            or
            epoch % 5 == 0
        ):

            print(
                f"Epoch {epoch:03d} "
                f"Loss={train_loss:.4f} "
                f"ValR2={val_r2:.4f} "
                f"ValMAE={val_mae:.4f} "
                f"ValRMSE={val_rmse:.4f}"
            )


        if val_mae < best_val_mae:

            best_val_mae = (
                val_mae
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0


        else:

            patience_counter += 1


        if patience_counter >= PATIENCE:

            print()
            print(
                "Early stopping at epoch:",
                epoch,
            )

            break


    model.load_state_dict(
        best_state
    )


    (
        test_r2,
        test_mae,
        test_rmse,
    ) = evaluate(
        model,
        test_loader,
    )


    print()
    print(
        "Best validation epoch:",
        best_epoch,
    )


    print(
        "Best validation MAE:",
        round(
            best_val_mae,
            4,
        ),
        "V",
    )


    print()
    print(
        "Final unseen-framework test:"
    )


    print(
        "R2:",
        round(
            test_r2,
            4,
        )
    )


    print(
        "MAE:",
        round(
            test_mae,
            4,
        ),
        "V",
    )


    print(
        "RMSE:",
        round(
            test_rmse,
            4,
        ),
        "V",
    )


    print()
    print(
        "Baselines:"
    )


    print(
        "GCN Fold 1 best observed R2: 0.3415"
    )


    print(
        "XGBoost Group CV mean R2: 0.4935"
    )


    print(
        "XGBoost Group CV mean MAE: 0.4185 V"
    )


    print()
    print(
        "STEP 14 completed."
    )


if __name__ == "__main__":

    main()