from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Vector4f as Vector4fD

def transform(v, mat):
    v = Vector4fD(v.x, v.y, v.z, 1.)
    v = mat @ v
    v = Vector3fD(v.x, v.y, v.z) / FloatD(v.w)
    return v