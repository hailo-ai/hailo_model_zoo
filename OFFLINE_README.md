# Offline Readme

This document describes how to view the Hailo Model Zoo documentation locally.

### Prerequisites

Have a copy of the Hailo Model Zoo locally.
This can be done by either:

1. cloning the Hailo Model Zoo.
    ```
    git clone https://github.com/hailo-ai/hailo_model_zoo.git
    ```

2. Installing the Hailo software suite. The docker image provided includes all Model Zoo documentation. See the [developer-zone](https://hailo.ai/developer-zone/) for installation instructions.

## View The Documentation Offline

Model Zoo documentation is written in `markdown` format. Here are few examples for offline viewers.

### VSCode

VSCode has out of the box support for Markdown files - [Markdown Preview](https://code.visualstudio.com/docs/languages/markdown#_markdown-preview).

### Grip

The below describes local setup for linux, but can be adapted for other OS.

1. Install `grip`:
    ```
    pip install grip
    ```

2. Launch the grip server:
    ```
    cd <DOCUMENTATION_FOLDER>
    env "PATH=$PATH" grip 8080
    ```
    <DOCUMENTATION_FOLDER> is where OFFLINE_README.md resides:

- When using the Git repo: ``Main cloned folder``.
- When using .whl installation (part of the Hailo SW Suite): ``HAILO_VIRTUALENV_FOLDER/hailo_model_zoo_doc``


3. Open browser and enter the following url:  `http://localhost:8080`
