with open("src/domain_models/config.py") as f:
    content = f.read()

# Replace _validate_single_trusted_dir and project_root validation
# Replace _validate_env_key, _validate_env_value, validate_env_file_security
old_validate_single = """def _validate_single_trusted_dir(path: str) -> str | None:
    if ".." in path:
        msg = f"Path traversal characters not allowed in trusted_directory: {path}"
        raise ValueError(msg)

    if not Path(path).is_absolute():
        msg = f"trusted_directory must be absolute: {path}"
        raise ValueError(msg)

    try:
        st_info = os.lstat(path)
        if stat.S_ISLNK(st_info.st_mode):
            msg = f"trusted_directory {path} cannot be a symlink."
            raise ValueError(msg)
    except FileNotFoundError:
        logging.warning(f"Trusted directory skipped as it does not exist: {path}")
        return None

    resolved = Path(path).resolve(strict=True)

    if not resolved.is_absolute():
        msg = f"trusted_directory must be absolute: {path}"
        raise ValueError(msg)
    if not resolved.is_dir():
        msg = f"trusted_directory must be a directory: {path}"
        raise ValueError(msg)

    from src.domain_models.config import SystemConfig

    restricted_prefixes = SystemConfig.model_fields["restricted_directories"].default_factory()  # type: ignore
    for restricted in restricted_prefixes:
        if str(resolved).startswith(restricted):
            msg = f"trusted_directory cannot be a system directory: {restricted}"
            raise ValueError(msg)

    st = resolved.stat()
    if bool(st.st_mode & stat.S_IWOTH):
        msg = f"trusted_directory {path} is world-writable, which is insecure."
        raise ValueError(msg)

    if st.st_uid != os.getuid():
        msg = f"trusted_directory {path} is not owned by the current user. This is insecure."
        raise ValueError(msg)

    return str(resolved)"""

# I will write a unified secure path resolver and validator
new_secure_path = """def _secure_resolve_and_validate_dir(path_str: str, check_exists: bool = True) -> str | None:
    path = Path(path_str)

    if not path.is_absolute():
        msg = f"Directory path must be absolute: {path_str}"
        raise ValueError(msg)

    if check_exists:
        try:
            st_info = os.lstat(path)
            if stat.S_ISLNK(st_info.st_mode):
                msg = f"Directory {path_str} cannot be a symlink."
                raise ValueError(msg)
        except FileNotFoundError:
            logging.warning(f"Directory skipped as it does not exist: {path_str}")
            return None

    resolved = path.resolve(strict=check_exists)

    if check_exists and not resolved.is_dir():
        msg = f"Path must be a directory: {path_str}"
        raise ValueError(msg)

    restricted_prefixes = ["/etc", "/bin", "/usr", "/sbin", "/var", "/lib", "/boot", "/root"]
    for restricted in restricted_prefixes:
        if str(resolved).startswith(restricted):
            msg = f"Directory cannot be a system directory: {restricted}"
            raise ValueError(msg)

    if check_exists:
        st = resolved.stat()
        if bool(st.st_mode & stat.S_IWOTH):
            msg = f"Directory {path_str} is world-writable, which is insecure."
            raise ValueError(msg)

        if st.st_uid != os.getuid():
            msg = f"Directory {path_str} is not owned by the current user. This is insecure."
            raise ValueError(msg)

    return str(resolved)"""

content = content.replace(old_validate_single, new_secure_path)

content = content.replace("""    @field_validator("project_root")
    @classmethod
    def validate_project_root_str(cls, v: str) -> str:
        if not v:
            msg = "project_root cannot be empty"
            raise ValueError(msg)

        if ".." in v:
            msg = "Path traversal sequences (..) are not allowed in project_root"
            raise ValueError(msg)

        resolved = Path(v).resolve(strict=True)
        if not resolved.is_absolute():
            msg = "project_root must be absolute"
            raise ValueError(msg)
        if not resolved.exists() or not resolved.is_dir():
            msg = "project_root must be an existing directory"
            raise ValueError(msg)

        # Security: forbid system directories
        from src.domain_models.config import SystemConfig

        restricted_prefixes = SystemConfig.model_fields["restricted_directories"].default_factory()  # type: ignore
        for restricted in restricted_prefixes:
            if str(resolved).startswith(restricted):
                msg = f"project_root cannot be a system directory: {restricted}"
                raise ValueError(msg)

        return str(resolved)""", """    @field_validator("project_root")
    @classmethod
    def validate_project_root_str(cls, v: str) -> str:
        res = _secure_resolve_and_validate_dir(v, check_exists=True)
        if res is None:
            msg = f"project_root skipped or invalid: {v}"
            raise ValueError(msg)
        return res""")

content = content.replace("""    @classmethod
    def validate_trusted_directories(cls, v: list[str]) -> list[str]:
        validated = []
        for path in v:
            res = _validate_single_trusted_dir(path)
            if res:
                validated.append(res)
        return validated""", """    @classmethod
    def validate_trusted_directories(cls, v: list[str]) -> list[str]:
        validated = []
        for path in v:
            res = _secure_resolve_and_validate_dir(path, check_exists=True)
            if res:
                validated.append(res)
        return validated""")

content = content.replace("""    @field_validator("trusted_directories")
    @classmethod
    def validate_trusted_directories(cls, v: list[str]) -> list[str]:
        validated = []
        for path in v:
            res = _validate_single_trusted_dir(path)
            if res:
                validated.append(res)
        return validated""", """    @field_validator("trusted_directories")
    @classmethod
    def validate_trusted_directories(cls, v: list[str]) -> list[str]:
        validated = []
        for path in v:
            res = _secure_resolve_and_validate_dir(path, check_exists=True)
            if res:
                validated.append(res)
        return validated""")


# Also need to replace the classmethod _validate_single_trusted_dir in DynamicsConfig that was left behind
import re

content = re.sub(r'    @classmethod\n    def _validate_single_trusted_dir[\s\S]*?return str\(resolved\)\n\n', '', content)


with open("src/domain_models/config.py", "w") as f:
    f.write(content)
