from pathlib import Path

BATCH_SIZE = 8
MODEL_NAME = '/from_s3/e5-mistral-7b-instruct'
DEVICE = 'cuda:0'
NUM_EPOCHS = 1
EVAL_STEPS = 64
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-4
OUTPUT_DIR = Path('/app/res')
