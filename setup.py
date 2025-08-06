import os
import shutil
import sys
import warnings
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info
from setuptools.command.sdist import sdist


class CustomEggInfo(egg_info):
    """Custom egg_info command that adds MIM extension files."""
    def run(self):
        # Create MIM files BEFORE running egg_info
        self.add_mim_extension()
        super().run()

    def add_mim_extension(self):
        """Add extra files that are required to support MIM into the
        package."""
        filenames = [
            'tools', 'configs', 'model-index.yml', 'dataset-index.yml'
        ]
        repo_path = os.path.dirname(os.path.abspath(__file__))
        mim_path = os.path.join(repo_path, 'mmpretrain', '.mim')

        print(f'Creating MIM files in {mim_path}', file=sys.stderr)
        os.makedirs(mim_path, exist_ok=True)

        for filename in filenames:
            if self._file_exists(filename):
                src_path = os.path.join(repo_path, filename)
                tar_path = os.path.join(mim_path, filename)

                # Remove existing file/directory
                if os.path.exists(tar_path):
                    if os.path.isfile(tar_path) or os.path.islink(tar_path):
                        os.remove(tar_path)
                    elif os.path.isdir(tar_path):
                        shutil.rmtree(tar_path)

                # Copy the file/directory
                try:
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, tar_path)
                        print(f'Copied file: {filename}')
                    elif os.path.isdir(src_path):
                        shutil.copytree(src_path, tar_path, dirs_exist_ok=True)
                        print(f'Copied directory: {filename}')
                except Exception as e:
                    warnings.warn(f'Failed to copy {filename}: {e}')

    def _file_exists(self, filename):
        """Check if file exists in the repository root."""
        repo_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.exists(os.path.join(repo_path, filename))


class CustomBuildPy(build_py):
    """Custom build_py command that ensures MIM files are included."""
    def run(self):
        # Ensure MIM files exist before building
        custom_egg_info = CustomEggInfo(self.distribution)
        custom_egg_info.add_mim_extension()
        super().run()


class CustomSdist(sdist):
    """Custom sdist command that ensures MIM files are included in source
    distribution."""
    def run(self):
        # Ensure MIM files exist before creating source distribution
        custom_egg_info = CustomEggInfo(self.distribution)
        custom_egg_info.add_mim_extension()
        super().run()


if __name__ == '__main__':
    setup(
        cmdclass={
            'egg_info': CustomEggInfo,
            'build_py': CustomBuildPy,
            'sdist': CustomSdist,
        })
