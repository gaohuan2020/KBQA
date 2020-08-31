def CheckKeyInConfig(key, config, type):
    if key in config:
        if type != None or type != "":
            if isinstance(config[key], type):
                return True
    return False