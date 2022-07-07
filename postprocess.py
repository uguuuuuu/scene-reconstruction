import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import argparse
import json
from util import prepare_for_mesh_opt, dump_opt_mesh

parser = argparse.ArgumentParser(description='Post-process some optimization results')

parser.add_argument('stage', type=int, help='The stage whose result to process, with value 0 or 1 meaning the first or second optimization stage.')
parser.add_argument('--config', type=str, help='Path to the config file')
parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint')

args = parser.parse_args()

FLAGS = json.load(open(args.config, 'r'))
ckp = args.checkpoint

if args.stage == 0:
    tet_res, tet_scale = FLAGS['tet_res'], FLAGS['tet_scale']
    prepare_for_mesh_opt(ckp, tet_res, tet_scale)
elif args.stage == 1:
    opt_dir = f"output/{FLAGS['name']}/optimized"
    dump_opt_mesh(ckp, opt_dir)