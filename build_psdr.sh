if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "The setpath.sh script must be sourced, not executed. In other words, run\n"
    echo "$ source setpath.sh\n"
    exit 0
fi

if [ "$BASH_VERSION" ]; then
    RECON_DIR=$(dirname "$BASH_SOURCE")
    export RECON_DIR=$(builtin cd "$RECON_DIR"; builtin pwd)
elif [ "$ZSH_VERSION" ]; then
    export RECON_DIR=$(dirname "$0:A")
fi

cd $RECON_DIR/ext/enoki
if [ ! -d "build" ]; then
    mkdir build
    cd build

    cmake -DENOKI_CUDA=ON -DENOKI_AUTODIFF=ON -DENOKI_PYTHON=ON ..
    make -j
fi



cd $RECON_DIR/ext/psdr-cuda

if [ ! -d "ext/optix" ]; then
    ln -s $1 ext/optix
fi

if [ ! -d "build" ]; then
    mkdir build
    cd build

    cmake -DENOKI_DIR="$RECON_DIR/ext/enoki" -DPYTHON_INCLUDE_PATH=$2 ..
    make -j
fi

cd $RECON_DIR