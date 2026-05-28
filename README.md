# M2PIM-project
A multimodal physics- and physiology-informed model (M2PIM) for accurate and robust dynamic beat-to-beat blood pressure estimation. This folder provides demo data and example code for dynamic beat-to-beat blood pressure estimation.

## Files

- `data/demo_subject_exercise_recovery.csv`: demo beat-level subject data with rest, exercise, and recovery segments.
- `main.py`: example implementation for SBP and DBP estimation.
- `requirements.txt`: required Python packages.

## Run

Install the required packages, then run:

```bash
python main.py
```

The script trains separate personalized models for SBP and DBP and saves the estimations, metrics, and plots to `results/`.
