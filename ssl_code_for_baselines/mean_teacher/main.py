import wandb
import torch
from dataset import dataloader_setup
from mt_utils import *
from trainer import train_one_epoch

def main(sweep=False):
    global global_step

    # If wandb.config is set, we read from it
    # Otherwise, we default to a manual run
    if sweep:
        print(f'IN SWEEP FUNCTION sweep = {sweep}')

        # If launched by wandb agent, wandb.run is not None
        batch_size = 32
        labeled_batch_size = 8
        epochs = 15
        consistency_rampup = 5
        cons_style ='sigmoid'

        import yaml
        with open("./sweep.yaml") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        run = wandb.init(config=config, project="mean-teacher-segmentation")

        # Note that we define values from `wandb.config`
        # instead of  defining hard values
        learning_rate = wandb.config.learning_rate
        consistency = wandb.config.consistency_weight
        thresh = wandb.config.cons_thresh
        alpha = wandb.config.alpha


    else:
        # Normal run if we do: python main.py
        alpha = 0.99
        batch_size = 32
        labeled_batch_size = 8
        epochs = 5
        learning_rate = 0.0005
        cons_style = 'sigmoid'
        consistency_rampup = 5
        consistency = 10
        thresh = 5


    print("==> Preparing data...")
    train_loader = dataloader_setup(batch_size, labeled_batch_size)

    print("==> Building model...")
    model, model_ema = build_models()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    supervised_losses = []
    consistency_losses = []
    consistency_weights = []
    print("==> Training model...")

    for epoch in range(epochs):
        cl_loss, cons_loss, cons_wt = train_one_epoch(
            train_loader=train_loader,
            model=model,
            ema_model=model_ema,
            optimizer=optimizer,
            epoch=epoch,
            batch_size=batch_size,
            alpha=alpha,
            total_epochs=epochs,
            visualize=False,
            cons_style=cons_style,
            consistency_rampup=consistency_rampup,
            consistency=consistency,
            sweep = sweep,
            thresh = thresh
        )
        supervised_losses.append(cl_loss)
        consistency_losses.append(cons_loss)
        consistency_weights.append(cons_wt)

    plot_train_metrics(range(1, epochs+1), supervised_losses, consistency_losses, consistency_weights)

    print("==> Saving models...")
    torch.save(model.state_dict(), "final_model.pth")
    torch.save(model_ema.state_dict(), "final_model_ema.pth")

if __name__ == "__main__":
    # Just call main() normally
    sweep=True
    main(sweep)
