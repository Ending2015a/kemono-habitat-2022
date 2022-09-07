# --- built in ---
import argparse
# --- 3rd party ---
# --- my module ---

def get_args():
  parser = argparse.ArgumentParser(
    description="Kemono agent"
  )
  parser.add_argument('--config', type=str, default='./Kemono/kemono_config.yaml')
  parser.add_argument('--evaluation', type=str, required=True, choices=['local', 'remote'])
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('dot_list', nargs=argparse.REMAINDER)
  args = parser.parse_args()
  return args