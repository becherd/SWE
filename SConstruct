#! /usr/bin/python

# @file
# This file is part of SWE.
#
# @author Alexander Breuer (breuera AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Dipl.-Math._Alexander_Breuer)
# @author Sebastian Rettenberger (rettenbs AT in.tum.de, http://www5.in.tum.de/wiki/index.php/Sebastian_Rettenberger,_M.Sc.)
#
# @section LICENSE
#
# SWE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SWE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SWE.  If not, see <http://www.gnu.org/licenses/>.
#
#
# @section DESCRIPTION
#
# Builds the SWE code with several options.
#

# print the welcome message
print '****************************************'
print '** Welcome to the build script of SWE **'
print '****************************************'
print 'SWE Copyright (C) 2012-2013'
print ''
print '  Technische Universitaet Muenchen'
print '  Department of Informatics'
print '  Chair of Scientific Computing'
print '  http://www5.in.tum.de/SWE'
print ''
print 'SWE comes with ABSOLUTELY NO WARRANTY.'
print 'SWE is free software, and you are welcome to redistribute it'
print 'under certain conditions.'
print 'Details can be found in the file \'gpl.txt\'.'
print ''

import os
import sys

#
# set possible variables
#
vars = Variables()

# read parameters from a file if given
vars.AddVariables(
  PathVariable( 'buildVariablesFile', 'location of the python file, which contains the build variables', None, PathVariable.PathIsFile )
)
env = Environment(variables=vars)
if 'buildVariablesFile' in env:
  vars = Variables(env['buildVariablesFile'])

# SWE specific variables
vars.AddVariables(
  PathVariable( 'buildDir', 'where to build the code', 'build', PathVariable.PathIsDirCreate ),

  EnumVariable( 'compiler', 'used compiler', 'gnu',
                allowed_values=('gnu', 'intel')
              ),

  EnumVariable( 'compileMode', 'mode of the compilation', 'release',
                allowed_values=('debug', 'analysis', 'release')
              ),

  EnumVariable( 'parallelization', 'level of parallelization', 'none',
                allowed_values=('none', 'openmp', 'opencl', 'cuda', 'mpi_with_cuda', 'mpi')
              ),

  EnumVariable( 'computeCapability', 'optional architecture/compute capability of the CUDA card', 'sm_20',
                allowed_values=('sm_10', 'sm_11', 'sm_12','sm_13',
                                'sm_20', 'sm_21', 'sm_22', 'sm_23' )
              ),

  BoolVariable( 'openGL', 'compile with OpenGL visualization', False),

  BoolVariable( 'openGL_instr', 'add instructions to openGL version (requires SDL_ttf)', False ),

  BoolVariable( 'writeNetCDF', 'write output in the netCDF-format', False ),

  BoolVariable( 'disableNonUniformNetCDFCells', 'always assume netcdf data cells are uniformly spaced', False ),

  BoolVariable( 'asagi', 'use ASAGI', False ),

  PathVariable( 'asagiInputDir', 'location of netcdf input files', '', PathVariable.PathAccept ),

  EnumVariable( 'solver', 'Riemann solver', 'dimsplit',
                allowed_values=('rusanov', 'dimsplit', 'fwave', 'augrie', 'hybrid', 'fwavevec')
              ),
                  
  BoolVariable( 'vectorize', 'add pragmas to help vectorization (release only)', False ),
                  
  BoolVariable( 'showVectorization', 'show loop vectorization (Intel compiler only)', False ),
  
  BoolVariable( 'showOptimization', 'show optimization report (Intel compiler only)', False ),

  EnumVariable( 'platform', 'compile for a specific platform (Intel compiler only)', 'default',
                allowed_values=('default', 'mic' )
              ),

  BoolVariable( 'disableUnitTests', 'do not build unit test targets', False ),
  
  BoolVariable( 'useNetCDFCache', 'load full netcdf files into memory for faster access', False ),
  
  BoolVariable( 'openCLProfiling', 'enable profiling of OpenCL Events', False ),
  
  BoolVariable( 'xmlRuntime', 'use a xml-file for runtime parameters', False )
)

# external variables
vars.AddVariables(
  PathVariable( 'cudaToolkitDir', 'location of the CUDA toolkit', None ),
  PathVariable( 'libSDLDir', 'location of libSDL', None),
  PathVariable( 'netCDFDir', 'location of netCDF', None),
  PathVariable( 'asagiDir', 'location of ASAGI', None),
  PathVariable( 'libxmlDir', 'location of libxml2', None),
  PathVariable( 'openCLIncludeDir', 'location of OpenCL includes', None),
  PathVariable( 'openCLLibDir', 'location of OpenCL libraries', None)
)

# set environment
env = Environment(ENV = {'PATH': os.environ['PATH'], 'LD_LIBRARY_PATH': os.getenv('LD_LIBRARY_PATH')},
        variables=vars,
        toolpath=['#submodules/cxxtest/build_tools/SCons'],
        tools = ['default', 'rightnow', ('cxxtest', {'CXXTEST_INSTALL_DIR':'#/submodules/cxxtest'})])

# generate help text
Help(vars.GenerateHelpText(env))

# handle unknown, maybe misspelled variables
unknownVariables = vars.UnknownVariables()

# remove the buildVariablesFile from the list of unknown variables (used before)
if 'buildVariablesFile' in unknownVariables:
  unknownVariables.pop('buildVariablesFile')

# exit in the case of unknown variables
if unknownVariables:
  print >> sys.stderr, "*** The following build variables are unknown:", unknownVariables.keys()
  Exit(1)


# valid solver for CUDA?
if env['parallelization'] in ['cuda', 'mpi_with_cuda'] and env['solver'] != 'rusanov' and env['solver'] != 'fwave':
  print >> sys.stderr, '** The "'+env['solver']+'" solver is not supported in CUDA.'
  Exit(3)

# CUDA parallelization for openGL
if env['parallelization'] != 'cuda' and env['openGL'] == True:
  print >> sys.stderr, '** The parallelization "'+env['parallelization']+'" does not support OpenGL visualization (CUDA only).'
  Exit(3)

# OpenMP parallelization for DimensionalSplitting
if env['parallelization'] in ['openmp', 'opencl'] and env['solver'] != 'dimsplit':
  print >> sys.stderr, '** The "'+env['solver']+'" solver is not supported in "'+env['parallelization']+'"".'
  Exit(3)

#
# precompiler, compiler and linker flags
#

# Select the compiler (MPI and/or Intel, GNU is default)
if env['parallelization'] in ['mpi', 'mpi_with_cuda']: 
  env['CXX'] = env['LINKERFORPROGRAMS'] = env.Detect(['mpiCC', 'mpicxx'])
  if not env['CXX']:
      print >> sys.stderr, '** MPI compiler not found, please update PATH environment variable'
      Exit(1)
  
  if env['compiler'] == 'intel':
    # We need to the the mpiCC wrapper which compiler it should use
    # Here are several environment variables that do the job for different MPI libraries
    envVars = ['OMPI_CXX', 'MPICH_CXX']
    for var in envVars:
      env['ENV'][var] = 'icpc'
elif env['parallelization'] == 'openmp':
    env.Append(CPPDEFINES=['USEOPENMP'])
    if env['compiler'] == 'intel':
        env.Append(CPPFLAGS=['-openmp'], LINKFLAGS=['-openmp'])
        env['CXX'] = 'icpc'
    else:
        env.Append(CPPFLAGS=['-fopenmp'], LINKFLAGS=['-fopenmp'])
else:
  if env['compiler'] == 'intel':
    env['CXX'] = 'icpc'

# eclipse specific flag
env.Append(CCFLAGS=['-fmessage-length=0'])

# xml parameters for the compiler TODO

# set (pre-)compiler flags for the compile modes
if env['compileMode'] == 'debug':
  env.Append(CPPDEFINES=['DEBUG'])
  env['openCLProfiling'] = True
  
  if env['compiler'] == 'gnu':
    env.Append(CCFLAGS=['-O0','-g3','-Wall'])

  elif env['compiler'] == 'intel':
    env.Append(CCFLAGS=['-O0','-g'])

elif env['compileMode'] == 'release':
  env.Append(CPPDEFINES=['NDEBUG'])

  if env['compiler'] == 'gnu':
    env.Append(CCFLAGS=['-O3','-mtune=native'])

  elif env['compiler'] == 'intel':
    env.Append(CCFLAGS=['-O2', '-xhost'])
elif env['compileMode'] == 'analysis':
  env['openCLProfiling'] = True
  env.Append(CPPDEFINES=['NDEBUG'])
  env.Append(CCFLAGS=['-fno-inline'])
  if env['compiler'] == 'gnu':
    env.Append(CCFLAGS=['-O3', '-g','-mtune=native'])

  elif env['compiler'] == 'intel':
    env.Append(CCFLAGS=['-O2', '-g', '-xhost'])
    
# Other compiler flags (for all compilers)
env.Append(CCFLAGS=['-fstrict-aliasing', '-fargument-noalias'])

# Vectorization?
if env['compileMode'] == 'release' and env['vectorize']:
  env.Append(CPPDEFINES=['VECTORIZE'])
  if env['compiler'] == 'intel':
    env.Append(CCFLAGS=['-xHost'])
if env['compiler'] == 'intel' and env['showVectorization']:
  env.Append(CCFLAGS=['-vec-report3'])
if env['compiler'] == 'intel' and env['showOptimization']:
  env.Append(CCFLAGS=['-opt-report'])
  
# Platform
if env['compiler'] == 'intel' and env['platform'] == 'mic':
  env.Append(CCFLAGS=['-mmic'])
  env.Append(LINKFLAGS=['-mmic'])
  
# Compiler
if env['compiler'] == 'intel':
  # Add Intel specific libraries
  env.Append(LIBS=['svml', 'imf', 'intlc'])
  
# Add source directory to include path (important for subdirectories)
env.Append(CPPPATH=['.'])

# set the precompiler variables for the solver
if env['solver'] == 'fwave':
  env.Append(CPPDEFINES=['WAVE_PROPAGATION_SOLVER=1'])
elif env['solver'] == 'augrie':
  env.Append(CPPDEFINES=['WAVE_PROPAGATION_SOLVER=2'])
elif env['solver'] == 'hybrid':
  env.Append(CPPDEFINES=['WAVE_PROPAGATION_SOLVER=0'])
elif env['solver'] == 'fwavevec':
  env.Append(CPPDEFINES=['WAVE_PROPAGATION_SOLVER=4'])

if env['useNetCDFCache']:
    env.Append(CPPDEFINES=['NETCDF_CACHE'])

# set the precompiler flags for CUDA
if env['parallelization'] in ['cuda', 'mpi_with_cuda']:
  env.Append(CPPDEFINES=['CUDA'])
  
  # set the directories for the CudaTool
  if 'cudaToolkitDir' in env:
    env['CUDA_TOOLKIT_PATH'] = env['cudaToolkitDir']

  env.Tool('CudaTool', toolpath = ['.'])
  
  # set precompiler flag for nvcc
  env.Append(NVCCFLAGS=['-DCUDA'])

  # set the compute capability of the cuda compiler (needs to be set after the CudaTool
  env.Append(NVCCFLAGS=['--gpu-architecture='+env['computeCapability']])
  
  # Append the source directory to the include path
  env.Append(NVCCFLAGS=['-Isrc'])
  
  # compile explicitly with 64-bit on Mac OS X
  if env['PLATFORM'] == 'darwin':
    env.Append(NVCCFLAGS=' -m64')

# set the nvcc precompiler flags for MPI (CUDA)
if env['parallelization'] == 'mpi_with_cuda':
  env.Append(NVCCFLAGS=['-DUSEMPI'])

# set the precompiler flags for MPI (C++)
if env['parallelization'] in ['mpi_with_cuda', 'mpi']:
  env.Append(CPPDEFINES=['USEMPI'])

if env['parallelization'] == 'opencl':
  if 'openCLIncludeDir' in env:
    env.Append(CPPPATH=[env['openCLIncludeDir']])
  if 'openCLLibDir' in env:
    env.Append(LIBPATH=[env['openCLLibDir']])
  
  env.Append(CPPDEFINES=['USEOPENCL'])
  # include dir for generated kernel files
  env.Append(CPPPATH=['blocks/opencl'])
  
  # TODO: allow linking against custom vendor
  # implementation (e.g. Nvidia CUDA) on Mac OS (Darwin)
  if env['PLATFORM'] == 'darwin':
    env.Append(CCFLAGS='-framework OpenCL')
    env.Append(LINKFLAGS='-framework OpenCL')
  else:
    env.Append(LIBS='OpenCL')

  if env['openCLProfiling']:
      env.Append(CPPDEFINES=['OPENCL_PROFILING'])

if env['openGL'] == True:
  env.Append(LIBS=['SDL', 'GL', 'GLU'])
  if env['openGL_instr'] == True:
    # We assume that SDL_ttf is in the same directory as SDL
    env.Append(LIBS=['SDL_ttf'])
    env.Append(CPPDEFINES=['USESDLTTF'])

# set the compiler flags for libSDL
if 'libSDLDir' in env:
  env.Append(CPPPATH=[env['libSDLDir']+'/include'])
  env.Append(LIBPATH=[env['libSDLDir']+'/lib'])
  env.Append(RPATH=[env['libSDLDir']+'/lib'])

# set the precompiler flags and includes for netCDF
if env['writeNetCDF'] == True:
  env.Append(CPPDEFINES=['WRITENETCDF'])
  env.Append(LIBS=['netcdf'])
  # set netCDF location
  if 'netCDFDir' in env:
    env.Append(CPPPATH=[env['netCDFDir']+'/include'])
    env.Append(LIBPATH=[os.path.join(env['netCDFDir'], 'lib')])
    env.Append(RPATH=[os.path.join(env['netCDFDir'], 'lib')])

# set the precompiler flags, includes and libraries for ASAGI
if env['asagi'] == True:
  env.Append(CPPDEFINES=['ASAGI'])
  if env['parallelization'] == 'none' or env['parallelization'] == 'cuda':
    env.Append(CPPDEFINES=['ASAGI_NOMPI'])
    env.Append(LIBS=['asagi_nompi'])
  else:
    env.Append(LIBS=['asagi'])
  if 'asagiDir' in env:
    env.Append(CPPPATH=[env['asagiDir']+'/include'])
    env.Append(LIBPATH=[env['asagiDir']+'/lib'])
    env.Append(RPATH=[os.path.join(env['asagiDir'], 'lib')])
  if 'netCDFDir' in env:
    env.Append(LIBPATH=[env['netCDFDir']+'/lib'])
    env.Append(RPATH=[os.path.join(env['netCDFDir'], 'lib')])
  if 'asagiInputDir' in env:
    env.Append(CPPFLAGS=['\'-DASAGI_INPUT_DIR="'+env['asagiInputDir']+'"\''])

# xml runtime parameters
if env['xmlRuntime'] == True: #TODO
  print 'xml runtime parameters are not implemented so far.'
  Exit(1)
  env.Append(CPPDEFINES=['READXML'])
  #set xmllib2 location
  if 'libxmlDir' in env:
    env.Append(CPPPATH=[env['libxmlDir']+'/include/libxml2'])
    env.Append(LIBPATH=[env['libxmlDir']+'/lib'])

# Disables the check and search for non-uniformly spaced cells
if env['disableNonUniformNetCDFCells'] == True:
  env.Append(CPPDEFINES=['DISABLE_NONUNIFORM_NETCDF_CELLS'])

#
# setup the program name and the build directory
#
program_name = 'SWE'

# compiler
program_name += '_'+env['compiler']

# compile mode
program_name += '_'+env['compileMode']

# parallelization
program_name += '_'+env['parallelization']

# solver
program_name += '_'+env['solver']

# build directory
build_dir = env['buildDir']+'/build_'+program_name

# get the src-code files
env.src_files = []
env.kernel_files = []
Export('env')
SConscript('src/SConscript', variant_dir=build_dir, duplicate=0)
Import('env')

# prepare opencl kernels for inclusion into program
if env['parallelization'] == 'opencl':
  # Transform OpenCL source into hex array
  oclBuilder = Builder(
      action = 'xxd -i $SOURCE > $TARGET',
      src_suffix = '.cl',
      suffix = '.h',
      prefix = 'kernels/')
  env.Append(BUILDERS = {'OpenCLKernel' : oclBuilder})
  
  # build kernel headers immediately
  for i in env.kernel_files:
    env.RightNow(env.OpenCLKernel(i))
    
  # Create header including all transformed OpenCL Kernel files
  # The header also includes the function getKernelSources 
  # with which an kernel source object for OpenCL can be retrieved
  kernelHeaderInclude = ''
  kernelHeaderFunction = ''
  for i in env.kernel_files:
    kernelBaseName = os.path.basename(i.rstr())
    kernelVarName = i.rstr().replace('.', '_').replace('/', '_')
    kernelHeaderInclude += '#include "kernels/' + os.path.splitext(kernelBaseName)[0] + '.h"\n'
    kernelHeaderFunction += '    src.push_back(std::make_pair((char*)'+kernelVarName+', '+kernelVarName+'_len));\n'
  kernelHeader = kernelHeaderInclude
  kernelHeader += 'void getKernelSources(cl::Program::Sources &src) {\n'
  kernelHeader += kernelHeaderFunction
  kernelHeader += '}\n'
  
  kernelFile = open(build_dir + '/blocks/opencl/kernels/kernels.h', 'w')
  kernelFile.truncate()
  kernelFile.write(kernelHeader)
  kernelFile.close()
  
# build the program
env.Program('build/'+program_name, env.src_files)
