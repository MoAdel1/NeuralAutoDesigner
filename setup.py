'''
DESCRIPTION
-----------
The setup file is used to pack auto_design module into .whl file to be installed directly in other environments 

build command : python setup.py bdist_wheel
'''

#%% code imports
from shutil import rmtree 
from setuptools import find_packages, setup


#%% main code section
if(__name__=='__main__'):
    # specify pkg name
    pkg_name = "auto_design"
    # build the .whl file
    setup(name=pkg_name,
          description="Automatic machine learning pipeline designing",
          version="0.1",
          packages=find_packages(include=[pkg_name, "{}.*".format(pkg_name)]),
          include_package_data=True,
          zip_safe=False,
          install_requires=['pandas>=0.25.1',
                            'keras>=2.2.4',
                            'numpy>=1.16.5',
                            'deap>=1.3.0',
                            'scikit-learn>=0.21.2',
                            'tensorflow>=1.14.0'])
    # delete un-needed files and folders
    rmtree("build")
    rmtree("{}.egg-info".format(pkg_name))
    