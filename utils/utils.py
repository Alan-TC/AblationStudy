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