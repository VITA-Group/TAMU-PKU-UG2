import os
import sys


if __name__ == "__main__":
    os.system('python2 ./hdrnet/bin/run.py ./pretrained_models/local_laplacian/strong_1024/ '+sys.argv[0]+' '+ sys.argv[1])
