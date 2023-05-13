import os
from itertools import product


class CommandsBuilder:
    r"""
    Creates the outer-product of configurations to be executed.
    Returns a list with all the combinations.
    Here's an example:
    ```
    commands = (
        CommandsBuilder()
        .add("dataset", ["Power", "Kin8mn"])
        .add("split", [0, 1])
        .build()
    )
    ```
    Returns
    ```
    commands = [
        "python main.py with dataset=Power split=0;",
        "python main.py with dataset=Power split=1;",
        "python main.py with dataset=Kin8mn split=0;",
        "python main.py with dataset=Kin8mn split=1;",
    ]
    ```
    """
    command_template = "python main.py {config};"
    single_config_template = " --{key}={value}"

    def __init__(self) -> None:
        self.keys = []
        self.values = []

    def add(self, key, values):
        self.keys.append(key)
        self.values.append(values)
        return self

    def build(self):
        commands = []
        for args in product(*self.values):
            config = ""
            for key, value in zip(self.keys, args):
                config += self.single_config_template.format(key=key, value=value)
            command = self.command_template.format(config=config)
            commands.append(command)
        return commands
    

DATASETS = [
    "se",
    "matern",
    "weaklyperiodic",
    "mixture",
    "sawtooth",
]

SCORE_PARAM = [
    "preconditioned_k",
    "preconditioned_s",
    "none",
    "y0"
]

if __name__ == "__main__":
    NAME = "commands_sawtooth.txt"

    if os.path.exists(NAME):
        print("File to store script already exists", NAME)
        exit(-1)

    commands = []

    commands.extend(
        CommandsBuilder()
        .add("config.data.dataset", ["sawtooth"])
        .add("config.network.num_bidim_attention_layers", [2, 4, 6])
        .add("config.network.hidden_dim", [64, 128, 256, 512])
        .add("config.network.num_heads", [4, 8, 16, 32])
        .build()
    )

    with open(NAME, "w") as file:
        file.write("\n".join(commands))