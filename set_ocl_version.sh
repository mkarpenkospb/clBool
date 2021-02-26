# you may need report, so quartus version should be consistent to SDK
SDK_VERSION=$1
QARTUS_VERSION=$2

export ALTERAOCLSDKROOT=/opt/intelFPGA_pro/$SDK_VERSION/hld # hmm
export SOCEDS_ROOT=/opt/intelFPGA_pro/17.1/embedded # hmm not existing directory
export QSYS_ROOTDIR=/opt/intelFPGA_pro/$SDK_VERSION/qsys/bin
export LD_LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/7.5.0:/opt/intelFPGA_pro/$SDK_VERSION/hld/host/linux64/lib:/opt/intelFPGA_pro/17.1/hld/board/euler2/linux64/lib
export INTELFPGAOCLSDKROOT=/opt/intelFPGA_pro/$SDK_VERSION/hld
export PATH=/opt/intelFPGA_pro/$SDK_VERSION/hld/bin:/opt/intelFPGA_pro/$QARTUS_VERSION/quartus/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin
export QUARTUS_ROOTDIR_OVERRIDE=/opt/intelFPGA_pro/$QARTUS_VERSION/quartus
# bsp always 17
export AOCL_BOARD_PACKAGE_ROOT=/opt/intelFPGA_pro/17.1/hld/board/euler2
