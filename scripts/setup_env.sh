#! /usr/bin/env bash
# supposed to be executed at project root
set -ex


if [[ ! -d ./venv_hailo_rpi_examples ]]; then
    ./install.sh
fi

. ./setup_env.sh

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
requirements_file="./requirements.txt"
installed_requirements_file="${requirements_file}_INSTALLED"

if [[ ! -f "$installed_requirements_file" ]] || ! cmp -s "$requirements_file" "$installed_requirements_file";
then
    pip install -r "$requirements_file"
    cp "$requirements_file" "$installed_requirements_file"
fi
