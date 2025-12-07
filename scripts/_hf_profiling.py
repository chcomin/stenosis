""" IN CONSTRUCTION """

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments


# 1. Crie o Callback que conecta o Trainer ao Profiler
class PyTorchProfilerCallback(TrainerCallback):
    def __init__(self, profiler):
        self.profiler = profiler
    
    def on_step_end(self, args, state, control, **kwargs):
        # A cada passo de treino completado, avança o profiler
        self.profiler.step()

# --- Configuração do seu Script ---

# Defina o schedule (muito importante para não gerar logs gigantes)
# wait=1, warmup=1, active=3, repeat=2 -> Grava apenas alguns passos iniciais
prof_schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2)

# Configure o Profiler
with torch.profiler.profile(
    schedule=prof_schedule,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log_profiler"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    
    # 2. Instancie seu Trainer (como você já faz normalmente)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # ... outros argumentos ...
    )
    
    # 3. Adicione o Callback injetando o objeto 'prof'
    trainer.add_callback(PyTorchProfilerCallback(prof))
    
    # 4. Inicie o treino DENTRO do 'with'
    trainer.train()