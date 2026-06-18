<!--
@author HCMUT HPC Summer School Hackathon Organizers
@permission DO_NOT_EDIT
@note Source-of-truth file ownership map for the hackathon package.
-->

# File Ownership And Edit Policy

Teams should only edit files marked `TEAM_EDITABLE` or `TEAM_ADVANCED_EDITABLE`.

## TEAM_EDITABLE

Teams may edit these files during the hackathon:

- `configs/hackathon_workflow.json`
- `prompts/*.txt`
- `src/hackathon_workflow.py`

## TEAM_ADVANCED_EDITABLE

Teams may edit these only when they intentionally change the agent design:

- `src/agents.py`

## DO_NOT_EDIT

These files are reference workload, scoring input, cluster wrapper, or workflow infrastructure:

- `HACKATHON.md`
- `FILE_OWNERSHIP.md`
- `qa_public.json`
- `qa_private.json`
- `slurm/hackathon-parallel-workflow.slurm`
- `src/ollama_client.py`
- `src/parallel_runner.py`
- `src/submission_builder.py`
- `src/submit.py`
- `src/trace_logger.py`
- `evidence/repo/**`
- `evidence/results/*.csv`
- `results/hackathon_*`
- `logs/*`
- `repo.zip`

## Notes

- JSON files cannot safely use comment syntax, so `configs/hackathon_workflow.json` carries its annotation in a valid `_annotation` field.
- `qa_public.json` and `qa_private.json` are not annotated in-place because they are scoring/question data. The workflow strips answer/evidence fields before prompting agents.
- Slurm files keep the shebang as the first line; annotation starts immediately after it.
