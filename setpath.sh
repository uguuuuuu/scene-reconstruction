#
# Adapted from Mitsuba 2 https://github.com/mitsuba-renderer/mitsuba2
#
# This script adds necessary binaries to the current path.
# It works with both Bash and Zsh 
#
# NOTE: this script must be sourced and not run, i.e.
#    . setpath.sh        for Bash
#    source setpath.sh   for Zsh or Bash
#


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

export PYTHONPATH="$RECON_DIR/ext/large-steps-pytorch/ext/botsch-kobbelt-remesher-libigl/build:$PYTHONPATH"
export PYTHONPATH="$RECON_DIR/ext/enoki/build:$RECON_DIR/ext/psdr-cuda/build/lib:$PYTHONPATH"
export LD_LIBRARY_PATH="$RECON_DIR/ext/enoki/build:$LD_LIBRARY_PATH"