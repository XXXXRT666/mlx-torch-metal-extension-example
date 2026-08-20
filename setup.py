from importlib.metadata import version as package_version

from mlx import extension
from setuptools import setup
from torch.utils.cpp_extension import include_paths, library_paths


setup(
    packages=["mlx_torch_metal_extension"],
    ext_modules=[
        extension.MetalExtension(
            "mlx_torch_metal_extension._ext",
            sources=[
                "src/activation.cpp",
                "src/silu_and_mul.metal",
            ],
            include_dirs=include_paths(),
            library_dirs=library_paths(),
            libraries=["c10", "torch", "torch_cpu"],
            runtime_library_dirs=["@loader_path/../torch/lib"],
            extra_compile_args={
                "cxx": ["-O3", "-fvisibility=hidden"],
                "metal": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": extension.BuildExtension.with_options(generate_stubs=False)},
    package_data={"mlx_torch_metal_extension": ["*.metallib", "*.pyi", "py.typed"]},
    install_requires=[
        f"mlx=={package_version('mlx')}",
        f"torch=={package_version('torch')}",
    ],
    zip_safe=False,
)
