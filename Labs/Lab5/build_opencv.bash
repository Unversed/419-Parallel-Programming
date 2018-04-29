#!/bin/bash

# Author: Cody Kitchener
# Date Last Modified: January 19th, 2017
# Email: cikitche@calpoly.edu
#
# If you have any questiosn or issues, please email me at
# the provided email.

initial_setup() {
# Set up a directory for OpenCV executables
mkdir $HOME/bin
# Set up the directory we want the built libraries to be
# saved to.
mkdir $HOME/opencv_build
# Create a temporary source folder within /tmp/;
# we need to build in /tmp/ because it's the only place
# with enough room to compile and build the libraries.
# Attempting to make the file within your student directory
# will exceed your disk quota.
mkdir /tmp/opencv_source

cd /tmp/opencv_source
# Download the latest version of OpenCV.
# As of January 19th, 2017, the master branch currently has
# OpenCV 3.2; there is another branch for 2.4 if you want to
# use that version instead.
git clone https://github.com/opencv/opencv.git

mkdir /tmp/opencv_source/opencv/release
# Quick check to make sure we're in the /tmp/ folder
# we created.
echo "In directory: " && pwd
}

cmake_configure() {

cd /tmp/opencv_source/opencv/release
# There are many, many, MANY OpenCV variables we can define
# to customize the library the way we want to. The best way to
# get a full list is to run ccmake and find which ones you want
# on (or off).
#
# The library we are building will NOT have CUDA or its
# associated libraries installed, or BLAS libraries for AMD
# processors and/or graphics cards.
#
# Parallel libraries that are installed are:
#	1) pthreads
#	2) openmp
#
# The CUDA host compiler is set to the appropriate one in case if
# there is any need to use nvvc later on (it defaults to gcc, which
# will give you a mind boggling error linking to the OS kernel if you
# use any GPU functions).
#
# The Javba libraries are disabled, and a few SSE (SIMD extensions for Intel CPUs)
# are enabled.
#
# Finally, probably the most important properties we are setting:
#	1) the buld type (release is selected in this case)
#	2) the folder we will install the build to
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=$HOME/opencv_build \
-D EXECUTABLE_OUTPUT_PATH=$HOME/bin \
-D BUILD_DOCS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_opencv_java=OFF \
-D BUILD_TBB=OFF \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=OFF \
-D CUDA_HOST_COMPILER=/usr/bin/g++ \
-D ENABLE_SSE41=ON \
-D ENABLE_SSE42=ON \
-D ENABLE_SSSE3=ON \
-D WITH_CUBLAS=OFF \
-D WITH_CUDA=OFF \
-D WITH_PTHREADS=ON \
-D WITH_NVCUVID=OFF \
-D WITH_OPENCLAMDBLAS=OFF \
-D WITH_OPENCLAMDFFT=OFF \
-D WITH_OPENMP=ON \
-D WITH_TBB=OFF \
..
}

make_run() {

cd /tmp/opencv_source/opencv/release
# Export the necessary environment varialbes to ensure that
# CMake has the right paths.
export PATH=$HOME/bin:$PATH
export PKG_CONFIG_PATH=$HOME/opencv_build/lib/pkgconfig:$PKG_CONFIG_PATH
# Making with 28 threads - the 28 is associated with the Cal Poly Impact Lab computers,
# which have 14 cores, so 28 threads seems like a pretty safe number to use.
# Overall, the build should take anywhere between 15 - 30 minutes (if it is successful).
echo -e "\nMaking...\n"
make -j28
}

make_install() {

cd /tmp/opencv_source/opencv/release

echo -e "\nInstalling...\n"
make install
}

cleanup() {

cd /tmp/opencv_source/opencv/release

echo -e "\nCleaing...\n"
make clean

cd $HOME
rm -rf /tmp/opencv_source
}

# In case if you forget, update your path files appropriatly in
# your .mybashrc file. The approach taken here is somewhat naive,
# and assumes that you have not set any path variables in the pass.
# Additionally, there's an OPENCV_DIR environment variable set to
# to the CMake settings for OpenCV.
set_mybashrc() {

cd $HOME

sed -i '5i \' .mybashrc
sed -i '6i OPENCV_SHARED=$HOME/opencv_build/share/OpenCV' .mybashrc
sed -i '6i PKG_CONFIG_PATH=$HOME/opencv_build/lib/pkgconfig:$PKG_CONFIG_PATH' .mybashrc
sed -i '6i C_INCLUDE_PATH=$HOME/opencv_build/include:$C_INCLUDE_PATH' .mybashrc
sed -i '6i LD_LIBRARY_PATH=$HOME/opencv_build/lib:$LD_LIBRARY_PATH' .mybashrc
sed -i '6i PATH=$HOME/bin:$PATH' .mybashrc

}

initial_setup
cmake_configure
make_run
make_install
cleanup
set_mybashrc
