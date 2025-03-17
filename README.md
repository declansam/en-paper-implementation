# en-paper-implementation


## Folder Descriptions

- **`src/`**: This folder contains the core scripts for training the model.
  - `main_v3.py`: The main Python script used for training. 
    - The function `sampleRunMain(CONFIG)` generates the first figure.
    - The function `main(CONFIG)` generates the second figure.
  - `data_generation.py`: A script used to simulate N-body movement data.

<br>

- **`csv/`**: Stores pickle output files that track the training process.

    - `sample_results_v3_sample_x.pkl`: Training statistics when trained on `x` number of samples (using function `sampleRunMain(CONFIG)`).

    - `train_results_v3_sample_x.pkl`: Training statistics when trained on `x` number of samples (using function `main(CONFIG)`).

    - `train_results_v2.pkl`: Training results containing egnn_test_loss, egnn_test_loss, best_egnn_loss, best_gnn_loss, train_loss, val_loss and total_time. 

<br>

- **`plots/`**: Contains all the plots generated during training, including:
  - `mse_curves_v3.png`: Compares the performance of EGNN and regular GNN.
  - `training_curves_v2.png`: Shows the training progression for both models.
  - `training_curves_v3_sample_x.png`: Shows the training progress for individual `x` samples.

- **`logs/`**: 

    - `slurm_en3_main_v2_58210102.out`: Holds training logs, when run with `main(CONFIG)`.
    - `slurm_en3_main_v3_58225060.out`: Holds training logs, which contains the overall results from the training process, when executed with `sampleRunMain(CONFIG)` function. 
---

```
NOTE: I created two versions of `main_v3.py` file for parallelization in HPC - one to run main(CONFIG) function and the other to run sampleRunMain(CONFIG) function.
```

For further details, please check the logs and generated plots in their respective directories.
