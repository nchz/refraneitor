#!/bin/bash -e

project_name=$(basename `git config --get remote.origin.url` | sed 's/.git$//')
project_root_dir=$(git rev-parse --show-toplevel)

# whether to use or not GPU.
if [[ "$MODE" = "cpu" ]]; then
    docker_run='docker run'
    maybe_gpu=''
else
    docker_run='docker run --runtime=nvidia'
    maybe_gpu='-gpu'
fi

# parse arguments.
cmd=$1
shift

case ${cmd} in
    h | help)
        # self-print to stdout.
        cat $0 | less
        ;;

    b | build)
        docker build -t ${project_name} \
            --build-arg MAYBE_GPU=${maybe_gpu} \
            ${project_root_dir}
        ;;

    f | download-fasttext-model)
        ft_model_filename='cc.es.300.bin.gz'
        wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/${ft_model_filename} - | gunzip > ${project_root_dir}/fasttext-models/${ft_model_filename}
        ;;

    r | run)
        ${docker_run} --rm -it -p 8888:8888 \
            -v ${project_root_dir}/data:/data \
            -v ${project_root_dir}/src:/src \
            -v ${project_root_dir}/fasttext-models:/fasttext-models \
            ${project_name} "$@"
        ;;

    *)
        echo "Bad command. Options are:"
        grep -E "^    . \| .*\)$" $0
    ;;

esac
