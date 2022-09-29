import yaml
import pathlib


def get_relative_path(__file__) -> pathlib.Path:
    return pathlib.Path(__file__).parent.resolve()


class ConfigBase:

    @staticmethod
    def _get_config() -> dict:
        path = get_relative_path(__file__)
        path = path.parent.absolute()
        try:
            with open(path / 'config.yaml', 'r') as stream:
                return yaml.safe_load(stream)

        except FileNotFoundError:
            with open(path / 'config.yaml', 'r') as stream:
                return yaml.safe_load(stream)
    
    @staticmethod
    def _get_relative_path(__file__) -> pathlib.Path:
        return get_relative_path(__file__)
