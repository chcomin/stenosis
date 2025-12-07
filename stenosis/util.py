""" Utility functions for profiling code execution using PyTorch profiler."""

from contextlib import contextmanager
from pathlib import Path

from torch.profiler import ProfilerActivity, record_function, schedule
from torch.profiler import profile as pt_profile


class Profiler:
    """Profile the code using the Pytorch profiler. The profiler warms up for 2 steps and then
    profiles for `num_steps` steps.

    Example usage:
    profiler = Profiler()
    profiler.start("train")
    for img, target in data_loader:
        with profiler.section("data_loader"):
            ...
            model(img)
            ...
        profiler.step()
    """

    def __init__(
            self, 
            num_steps: int = 3,
            include_cuda: bool = True, 
            trace_path: str | Path = "./", 
            record_shapes: bool = True,
            with_stack: bool = True,
            enabled: bool = True
            ):
        """
        Parameters
        ----------
        num_steps
            Number of steps to profile.
        include_cuda
            If True, include CUDA activities in the profiling.
        trace_path
            Path to save the trace files.
        record_shapes
            If True, record shapes of the tensors.
        with_stack
            If True, record the stack trace.
        enabled
            If False, make all operations noop.
        """
        
        WARMUP_STEPS = 2

        activities = [ProfilerActivity.CPU]
        if include_cuda:
            activities += [ProfilerActivity.CUDA]

        profiler_schedule = schedule(
            skip_first=0,
            wait=0,
            warmup=WARMUP_STEPS,
            active=num_steps,
            repeat=1
            )
        
        self._pytorch_profiler_args = {
            "activities": activities,
            "schedule": profiler_schedule,
            "record_shapes": record_shapes,
            "profile_memory": True,
            "with_stack": with_stack
        }
        
        self.max_num_steps = WARMUP_STEPS + num_steps
        self.trace_path = Path(trace_path)
        self.enabled = enabled
        self.profiler = None
        self.active = False

    def start(self, trace_name: str = "trace"):
        """Start profiling. The trace file will be saved as {trace_name}.json."""

        if not self.enabled: 
            return
        
        trace_file = str(self.trace_path/f"{trace_name}.json")
        self.profiler = pt_profile(
            **self._pytorch_profiler_args,
            on_trace_ready=lambda x: x.export_chrome_trace(trace_file)
            )
        self.profiler.__enter__()
        self.active = True

    @contextmanager
    def section(self, section_name):
        """Context manager to mark a section of code for the profiler."""
        section = record_function(section_name)
        section.__enter__()
        try:
            yield 
        finally:
            section.__exit__(None, None, None)

    def step(self):
        """Step the profiler. If the number of steps is reached, the profiler is stopped."""
        if not self.enabled: 
            return
        if self.active:
            self.profiler.step()
            if self.profiler.step_num == self.max_num_steps:
                self.profiler.__exit__(None, None, None)
                self.active = False

    def is_enabled(self):
        return self.enabled