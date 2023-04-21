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

    keys = []
    values = []

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
    "sawtooth",
    "mixture",
]


if __name__ == "__main__":
    NAME = "commands_regression1d.txt"

    if os.path.exists(NAME):
        print("File to store script already exists", NAME)
        exit(-1)

    commands = (
        CommandsBuilder()
        .add("config.sde.is_score_precond", [True, False])
        .add("config.data.dataset", DATASETS)
        .add("config.sde.limiting_kernel", ["white", "se", "matern52"])
        .add("config.network.translation_invariant", [True, False])
        .build()
    )

    with open(NAME, "w") as file:
        file.write("\n".join(commands))