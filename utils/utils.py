from pathlib import Path

import toml


def read_config(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            config = toml.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except toml.TomlDecodeError as e:
        print(f"Error decoding TOML: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def validate_file(file_path):
    file = Path(file_path)
    if not file.is_file():
        raise Exception(f"Arquivo {file} não encontrado.")
    return file

def validate_folder(folder_path, create=True):
    folder = Path(folder_path)
    if not folder.is_dir():
        if create:
            folder.mkdir(parents=True, exist_ok=True)
        else:
            raise Exception(f"Pasta {folder} não encontrada.")
    return folder