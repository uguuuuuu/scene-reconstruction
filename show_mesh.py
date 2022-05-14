import potpourri3d as pp3d
import polyscope as ps
import argparse

parser = argparse.ArgumentParser('Load a mesh and display it')
parser.add_argument('path', type=str, help='path of the mesh')
args = parser.parse_args()

v, f = pp3d.read_mesh(args.path)

ps.init()
ps.register_surface_mesh('m', v, f)
ps.show()