#! /usr/bin/env bash

source ../setup_venv.sh

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
requirements_file="$script_dir/requirements.txt"
installed_requirements_file="${requirements_file}_INSTALLED"

if [[ ! -f "$installed_requirements_file" ]] || ! cmp -s "$requirements_file" "$installed_requirements_file";
then
    pip install -r "$requirements_file"
    cp "$requirements_file" "$installed_requirements_file"
fi
