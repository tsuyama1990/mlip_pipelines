import re

with open("src/trainers/finetune_manager.py", "r") as f:
    text = f.read()

replacement_mace = '''    def _run_mace_subprocess(
        self, mace_train_bin: str, train_xyz: Path, model_path: str, temp_dir: Path
    ) -> None:
        import re
        import subprocess
        import logging

        # Strictly validate all command components
        train_xyz_str = str(train_xyz.resolve(strict=True))
        if not re.match(r"^[/a-zA-Z0-9_.-]+$", train_xyz_str) or ".." in train_xyz_str:
            msg = f"Invalid characters in train_xyz path: {train_xyz_str}"
            raise ValueError(msg)

        if not re.match(r"^[/a-zA-Z0-9_.-]+$", model_path) or ".." in model_path:
            msg = f"Invalid characters in model_path: {model_path}"
            raise ValueError(msg)

        temp_dir_str = str(temp_dir.resolve(strict=True))
        if not re.match(r"^[/a-zA-Z0-9_.-]+$", temp_dir_str) or ".." in temp_dir_str:
            msg = f"Invalid characters in temp_dir path: {temp_dir_str}"
            raise ValueError(msg)

        if not isinstance(self.config.mace_finetuning_epochs, int) or self.config.mace_finetuning_epochs <= 0:
            msg = "Invalid mace_finetuning_epochs"
            raise ValueError(msg)

        if not isinstance(self.config.mace_learning_rate, float) or self.config.mace_learning_rate <= 0.0:
            msg = "Invalid mace_learning_rate"
            raise ValueError(msg)

        cmd = [
            mace_train_bin,
            "--train_file",
            train_xyz_str,
            "--model",
            model_path,
            "--output_dir",
            temp_dir_str,
        ]
        if self.config.mace_freeze_body:
            cmd.append("--freeze_body")
        cmd.extend(
            [
                "--max_num_epochs",
                str(self.config.mace_finetuning_epochs),
                "--lr",
                str(self.config.mace_learning_rate),
            ]
        )

        try:
            subprocess.run(  # noqa: S603
                cmd,
                check=True,
                capture_output=True,
                text=True,
                shell=False,
                timeout=self.config.timeout,
            )
        except subprocess.TimeoutExpired as e:
            logging.exception(
                f"mace_run_train execution timed out after {self.config.timeout} seconds."
            )
            msg = "mace_run_train execution timed out."
            raise RuntimeError(msg) from e'''

text = re.sub(
    r'    def _run_mace_subprocess\([\s\S]*?raise RuntimeError\(msg\) from e',
    replacement_mace,
    text, count=1
)

with open("src/trainers/finetune_manager.py", "w") as f:
    f.write(text)
