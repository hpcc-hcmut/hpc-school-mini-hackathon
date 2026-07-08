# Adaptation Notes

This assignment artifact is designed to be highly reusable across different contexts. Instructors can scale the assignment along three primary axes:

## 1. Content Axis
The current implementation uses a CIFAR-100 ResNet training repository. You can adapt the assignment to other domains by replacing the evidence corpus:
- CFD (Computational Fluid Dynamics) scripts and output logs.
- NWChem or molecular dynamics workflows.
- LLM inference or finetuning setups.
- General MPI/OpenMP workload repositories.
- Data analytics pipelines.

## 2. Workflow Axis
The complexity of the task can be adjusted based on student experience levels:
- **Beginner Mode:** Students only edit prompts and configurations (e.g., model routing, temperature) to improve baseline answers.
- **Standard Mode:** Students edit model routing, context sizes, and batching strategies to optimize cluster execution time and resource usage.
- **Advanced Mode:** Students redesign agent roles, implement complex verification logic, or rewrite parallel execution strategies in the core workflow logic.

## 3. Infrastructure Axis
The repository can run in several environments:
- **Full Cluster Setup:** Slurm + GPU nodes + local Apptainer LLM serving (Ollama).
- **Cloud/Hosted Setup:** Replacing the local Ollama client with hosted LLM endpoints (e.g., OpenAI API).
- **Container Variations:** Swapping Apptainer for Docker.
- **Local/Mock Execution:** Using the built-in mock mode for offline development, or utilizing offline traces for testing without GPUs.
