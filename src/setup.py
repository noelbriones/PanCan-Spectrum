import sys, os
from cx_Freeze import setup, Executable

# os.environ['TCL_LIBRARY']=r'C:/Programs/Python/Python35-32/tcl/tcl8.6'
# os.environ['TK_LIBRARY']=r'C:/Programs/Python/Python35-32/tcl/tk8.6'

import os.path

PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')

includefiles = ['SP_logo.png', 'SP_logo.ico', 'PDF_banner.png', 'temp', 'svm_models']
includes = []
excludes = ['scipy.spatial.cKDTree']
packages = ['matplotlib', 'scipy', 'fpdf', 'time', 'numpy', 'pymzml', 'math',
            'sklearn', 'pysimplegui', 'joblib', 'pymysql', 'statistics', 'tkinter', 'sys']
options = {
    'build_exe': {
        'include_files':[
            os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tk86t.dll'),
            os.path.join(PYTHON_INSTALL_DIR, 'DLLs', 'tcl86t.dll'),
            'SP_logo.png', 'SP_logo.ico', 'PDF_banner.png', 'temp', 'svm_models'
         ],
    },
}

setup(name="PanCan Spectrum",
      version="1.1",
      description="A Pancreatic Cancer Detection Support Tool Using Mass Spectrometry Data and Support Vector Machines",
      options = options,
      # options = {'build_exe': {'excludes':excludes,'packages':packages,'include_files':includefiles}},
      executables=[Executable("driver.py", base=None)]
      )
