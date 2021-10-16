import setuptools

# Developer self-reminder for uploading in pypi:
# - install: wheel, twine
# - build  : python setup.py bdist_wheel
# - deploy : twine upload dist/*

with open("README.md", "r") as file:
    long_description = file.read()

setuptools.setup(
    name='gnntf',
    version='0.0.9',
    author="Emmanouil (Manios) Krasanakis",
    author_email="maniospas@hotmail.com",
    description="Graph neural networks on tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maniospas/gnn-test",
    packages=setuptools.find_packages(),
    classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache Software License",
         "Operating System :: OS Independent",
     ],
    install_requires=[
              'sklearn', 'scipy', 'numpy', 'networkx', 'tensorflow'
      ],
 )