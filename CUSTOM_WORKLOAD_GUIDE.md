# Custom Workload Guide

Instructors wishing to adapt this hackathon to a new HPC workload should follow these steps:

1. **Replace the Core Repository:**
   Swap out `evidence/repo/` with your custom workload's source code, batch scripts, and configurations.

2. **Replace Benchmark Results:**
   Update `evidence/results/` with performance summaries, traces, or benchmark logs relevant to the new workload.

3. **Replace Public Sample Questions:**
   Update `qa_public.json` with scaffolding questions that guide students to explore the new evidence repository.

4. **Update the Instructor-Side Private Files:**
   In the instructor-facing leaderboard repository, replace the private questions and the ground truth scoring files to match the new domain.

5. **Update Default Prompts:**
   Adjust the baseline prompts in `prompts/` to better handle the specific terminology and structure of the new domain (e.g., if switching from PyTorch to MPI/C++).

6. **Pre-Class Validation:**
   - Test the Python mock mode locally.
   - Perform at least one full baseline Slurm run before the live event to ensure the Ollama serving stack and workflow run seamlessly on the new evidence corpus.
   - Validate a generated submission against the paired leaderboard's current
     `SubmissionRequest`, then exercise the authenticated submission API.

The mock check requires Python 3.10+ and a writable output directory only. The
full path additionally requires site-provided containers and a pre-staged Ollama
cache; container build recipes are intentionally not invented by this artifact.
