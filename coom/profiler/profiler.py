from lightning.pytorch.profilers import PyTorchProfiler
from torch.profiler import tensorboard_trace_handler, schedule
import os
from typing import Optional, Dict, Any


def get_default_profiler_config() -> Dict[str, Any]:
    """
    Get default profiler configuration.
    
    Returns:
        Dict[str, Any]: Default profiler configuration.
    """
    return {
        "schedule": {
            "wait": 2,
            "warmup": 2, 
            "active": 6,
        },
        "export_to_chrome": True,
        "export_to_tensorboard": True
    }


def create_profiler_schedule(wait: int = 2, warmup: int = 2, active: int = 6):
    """
    Create a profiler schedule based on the provided parameters.
    
    Args:
        wait (int): Number of steps to wait before starting profiling.
        warmup (int): Number of warmup steps.
        active (int): Number of active profiling steps.
        
    Returns:
        torch.profiler.schedule: Configured profiler schedule.
    """
    return schedule(wait=wait, warmup=warmup, active=active)


def pytorch_profiler(
    experiment_name: str, 
    sub_experiment_name: Optional[str] = None, 
    profiler_config: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    Create a PyTorch profiler with configurable scheduling.
    
    Args:
        experiment_name (str): Name of the experiment.
        sub_experiment_name (Optional[str]): Name of the sub-experiment.
        profiler_config (Optional[Dict[str, Any]]): Profiler configuration loaded from Hydra.
        **kwargs: Additional arguments to override config settings.
        
    Returns:
        PyTorchProfiler: Configured profiler instance.
    """
    # Use provided config or default
    config = profiler_config if profiler_config is not None else get_default_profiler_config()
    
    # Set up profiler directory
    if sub_experiment_name is not None:
        profiler_dir = os.path.join("profiler_logs", experiment_name, sub_experiment_name)
    else:
        profiler_dir = os.path.join("profiler_logs", experiment_name)

    os.makedirs(profiler_dir, exist_ok=True)

    # Extract schedule configuration
    schedule_config = config.get("schedule", {})
    wait = schedule_config.get("wait", 2)
    warmup = schedule_config.get("warmup", 2)
    active = schedule_config.get("active", 6)
    
    # Create the profiler schedule
    profiler_schedule = create_profiler_schedule(wait, warmup, active)
    
    # Set up profiler arguments
    profiler_kwargs = {
        "dirpath": profiler_dir,
        "filename": "profile",
        "schedule": profiler_schedule,
        "export_to_chrome": config.get("export_to_chrome", True),
        "export_to_tensorboard" : config.get("export_to_tensorboard", True),
    }
    
    
    # Override with any provided kwargs
    profiler_kwargs.update(kwargs)

    
    return PyTorchProfiler(**profiler_kwargs)