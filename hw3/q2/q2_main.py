import os
from q2.mesh import Mesh
import numpy as np


def main():
    src_dir = os.path.join('..', 'FAUST')
    files_list = [os.path.join(src_dir, f) for f in os.listdir(src_dir)]
    mesh = Mesh(files_list[0])
    a, b = mesh.laplacian_spectrum(k=5, cls='half_cotangent')
    print(a)


if __name__ == "__main__":
    main()
