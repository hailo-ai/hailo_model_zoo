ALLS_COMPLETE = {
    "bash": "_shtab_greeter_compgen_ALLSFiles",
    "zsh": "_files -g '(*.alls|*.ALLS)'",
    "tcsh": "f:*.alls",
}

CKPT_COMPLETE = {
    "bash": "_shtab_greeter_compgen_CKPTFiles",
    "zsh": "",
    "tcsh": "",
}
HAR_COMPLETE = {
    "bash": "_shtab_greeter_compgen_HARFiles",
    "zsh": "_files -g '(*.har|*.HAR)'",
    "tcsh": "f:*.har",
}
HEF_COMPLETE = {
    "bash": "_shtab_greeter_compgen_HEFFiles",
    "zsh": "_files -g '(*.hef|*.HEF)'",
    "tcsh": "f:*.hef",
}

JSON_COMPLETE = {
    "bash": "_shtab_greeter_compgen_JSONFiles",
    "zsh": "_files -g '(*.json|*.JSON)'",
    "tcsh": "f:*.json",
}

TFRECORD_COMPLETE = {
    "bash": "_shtab_greeter_compgen_TFRECORDFiles",
    "zsh": "_files -g '(*.tfrecord|*.TFRECORD)'",
    "tcsh": "f:*.tfrecord",
}

YAML_COMPLETE = {
    "bash": "_shtab_greeter_compgen_YAMLFiles",
    "zsh": "_files -g '(*.yaml|*.YAML)'",
    "tcsh": "f:*.yaml",
}

DEVICE_COMPLETE = {
    "bash": "_shtab_compgen_hailo_device",
    "zsh": "",
    "tcsh": "",
}

FILE_COMPLETE = {
    "bash": "_shtab_compgen_files",
    "zsh": "_files",
    "tcsh": "f",
}
