from sys import argv as args
from crater_deep_network import run_experiments
import os

def main():
  run_experiments()

if __name__ == "__main__":
  # Make sure crater pickle is there
  #os.system('rm Pickles/*.pkl')
  main()
