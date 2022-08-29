"""A little utility library to make for easy logging to wandb."""
import math
import secrets
import time
from typing import Optional
from typing import Union

import numpy as np
import torch
import wandb
from torch import Tensor
from wandb.sdk.wandb_run import Run

last_time: Optional[float] = None
last_step = 0
LOG_BACKOFF_POINT = 5000
LOG_BACKOFF_FACTOR = 20

# backoff does apply to scalars and media
SCALAR_FREQ = 5
MEDIA_FREQ = 100
# backoff does not apply to hists
HIST_FREQ = 100

# TODO: instead of every call including the step (and thus having to know the step in all code),
# we should just update the step once per iteration with a special call.
# Why didn't I do it that way in the first place? Does that limit any capabilities?


def effective_freq(step: int, freq: int) -> int:
    # Simple logging backoff logic.
    if step > LOG_BACKOFF_POINT and freq != 1:
        freq *= LOG_BACKOFF_FACTOR
    return freq


def check_log_interval(step: int, freq: int, backoff: bool = True) -> bool:
    if backoff:
        freq = effective_freq(step, freq)
    return step % freq == 0


def log_scalar(
    tag: str, value: Union[int, float, Tensor], step: Optional[int] = None, freq: Optional[int] = None
) -> None:
    freq = SCALAR_FREQ if freq is None else freq
    if step is None:
        step = last_step
    if not check_log_interval(step, freq):
        return
    if type(value) == torch.Tensor:
        value = value.cpu().detach()
    wandb.log({tag: value}, step=step)


def log_histogram(
    tag: str,
    value: Tensor,
    step: Optional[int] = None,
    mean_freq: Optional[int] = None,
    hist_freq: Optional[int] = None,
    log_mean: bool = True,
) -> None:
    mean_freq = SCALAR_FREQ if mean_freq is None else mean_freq
    hist_freq = HIST_FREQ if hist_freq is None else hist_freq

    if step is None:
        step = last_step

    # Allow backoff to apply to the scalars
    if log_mean:
        try:
            # works on torch and tf
            mean = value.mean()
        except AttributeError:
            # hopefully works on most everything else
            mean = np.mean(value)
        log_scalar(tag + "_mean", mean, step, mean_freq)
    # Backoff doesn't work well with hists, messes up the viz
    if not check_log_interval(step, hist_freq, backoff=False):
        return
    if type(value) == torch.Tensor:
        value = value.cpu().detach()
    wandb.log({tag + "_hist": wandb.Histogram(value)}, step=step)  # type: ignore


def log_video(
    tag: str,
    batch: Tensor,
    step: Optional[int] = None,
    freq: Optional[int] = None,
    normalize: bool = False,
    num_images_per_row: int = 4,
) -> None:
    freq = MEDIA_FREQ if freq is None else freq
    if step is None:
        step = last_step
    # Expects b, t, c, h, w
    # or t, c, h, w
    if not check_log_interval(step, freq):
        return

    if normalize:
        min_v = torch.min(batch)
        range_v = torch.max(batch) - min_v
        if range_v > 0:
            batch = (batch - min_v) / range_v
        else:
            batch = torch.zeros(batch.size())

    # batch = preprocess_batch(batch).permute(0, 2, 1, 3, 4) + .5
    if len(batch.shape) == 5:
        frames = make_video_grid(batch, num_images_per_row=num_images_per_row, pad_value=1)
    else:
        assert len(batch.shape) == 4
        frames = batch

    # This should be in range 0-1
    if type(frames) == torch.Tensor:
        frames = frames.detach()
        if torch.is_floating_point(frames):
            frames = (frames * 255).clamp(0, 255).to(torch.uint8)
        frames = frames.cpu()
    elif isinstance(frames, np.ndarray):
        if np.issubdtype(frames.dtype, np.floating):
            frames = (frames * 255).clip(0, 255).astype(np.uint8)
    else:
        assert False
    wandb.log({tag: wandb.Video(frames, fps=4, format="webm")}, step=step)  # type: ignore


def log_image(tag: str, value: Tensor, step: int, freq: Optional[int] = None) -> None:
    freq = MEDIA_FREQ if freq is None else freq
    if not check_log_interval(step, freq):
        return
    if type(value) == torch.Tensor:
        value = value.cpu().detach()
    wandb.log({tag: wandb.Image(value)}, step=step)


def log_table(tag: str, value: Tensor, step: int, freq: Optional[int] = None):
    freq = MEDIA_FREQ if freq is None else freq
    # value should be a 1d tensor, in this current implementation. can add more columns in the future.
    if not check_log_interval(step, freq):
        return
    if type(value) == torch.Tensor:
        value = value.cpu().detach()
    columns = ["test"]
    rows = [[x] for x in value]
    table = wandb.Table(data=rows, columns=columns)
    wandb.log({tag: table}, step=step)


def log_iteration_time(batch_size: int, step: int, freq: Optional[int] = None) -> None:
    """Batch size should be the number of samples processed per step of i."""
    freq = SCALAR_FREQ if freq is None else freq
    global last_time
    global last_step
    if not check_log_interval(step, freq):
        return

    if last_time is None:
        last_time = time.time()
        last_step = step
    else:
        if step == last_step:
            return
        dt = (time.time() - last_time) / (step - last_step)
        last_time = time.time()
        last_step = step
        log_scalar("timings/iterations-per-sec", 1 / dt, step, freq=1)
        log_scalar("timings/samples-per-sec", batch_size / dt, step, freq=1)


def watch(model: torch.nn.Module, freq: Optional[int] = None) -> None:
    freq = HIST_FREQ if freq is None else freq
    wandb.watch(model, log="all", log_freq=freq, log_graph=False)


def download_file(run_id: str, project_name: str, filename: Optional[str] = None) -> str:
    api = wandb.Api()
    run = api.run(f"sourceress/{project_name}/{run_id}")
    # We save to a random directory to avoid contention issues if there's
    # potentially multiple processes downloading the same file at the same time.
    # TODO: clean these files up
    path = run.file(filename).download(replace=True, root=f"./data/{secrets.token_hex(10)}").name
    assert isinstance(path, str)
    return path


def make_video_grid(
    tensor: Tensor,
    num_images_per_row: int = 10,
    padding: int = 2,
    pad_value: int = 0,
) -> Tensor:
    """
    This is a repurposed implementation of `make_grid` from torchvision to work with videos.

    Expects shape (batch_size, timesteps, channels, h, w).
    """
    n_maps, sequence_length, num_channels, height, width = tensor.size()
    x_maps = min(num_images_per_row, n_maps)
    y_maps = int(math.ceil(float(n_maps) / x_maps))
    height, width = int(height + padding), int(width + padding)
    grid = tensor.new_full(
        (sequence_length, num_channels, height * y_maps + padding, width * x_maps + padding), pad_value
    )
    k = 0
    for y in range(y_maps):
        for x in range(x_maps):
            if k >= n_maps:
                break
            grid.narrow(2, y * height + padding, height - padding).narrow(
                3, x * width + padding, width - padding
            ).copy_(tensor[k])
            k += 1
    return grid


def log_scalar_mp(wandb_run: Run, key: str, value: Union[int, float]) -> None:
    # Log a scalar from a separate process.
    # Must have initialized a new run from within the new process.
    try:
        wandb_run.log({key: value})
    except BrokenPipeError:
        # This error seems to happen when the main process shuts down before the child.
        print("wandb threw broken pipe error, continuing.")
