"""Repository guards for generated research data and build products."""

import os
import secrets
import stat
import subprocess
from pathlib import Path


class RepositoryHygieneError(ValueError):
    pass


GENERATED_ROOTS = {
    ".cache", "artifacts", "checkpoints", "data", "datasets", "downloads",
    "outputs", "runs",
}

FORBIDDEN_TRACKED_ROOTS = GENERATED_ROOTS.union({"build", "dist", "site"})

FORBIDDEN_TRACKED_SUFFIXES = (
    "$py.class", ".7z", ".arrow", ".bin", ".bz2", ".ckpt", ".class",
    ".cover", ".csv", ".dat", ".dll", ".dylib", ".egg", ".env",
    ".feather", ".h5", ".hdf5", ".json", ".listens", ".log", ".manifest",
    ".mat", ".mo", ".npy", ".npz", ".parquet", ".pkl", ".png", ".pot",
    ".pt", ".pth", ".pyc", ".pyd", ".pyo", ".so", ".spec", ".sqlite",
    ".sqlite3", ".tar", ".tar.bz2", ".tar.gz", ".tar.xz", ".tar.zst",
    ".tbz2", ".tgz", ".txt", ".txz", ".whl", ".xz", ".zip", ".zst",
)

FORBIDDEN_TRACKED_COMPONENTS = {
    ".cache", ".eggs", ".hypothesis", ".ipynb_checkpoints", ".mypy_cache",
    ".tox", ".venv", "__pycache__", "artifacts", "build", "checkpoints",
    "data", "datasets", "develop-eggs", "dist", "downloads", "eggs", "env",
    "htmlcov", "lib", "lib64", "outputs", "parts", "runs", "sdist",
    "site", "target", "var", "venv", "wheels",
}

FORBIDDEN_TRACKED_BASENAMES = {
    ".coverage", ".installed.cfg", "coverage.xml", "nosetests.xml",
    "pip-delete-this-directory.txt", "pip-log.txt",
}

ALLOWED_TRACKED_FIXTURES = {
    "examples/listenbrainz_synthetic.listens",
    "examples/tiny_events.dat",
}


def _nearest_existing(path):
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def repository_root(path):
    """Return the containing Git worktree, or None outside a worktree."""
    candidate = _nearest_existing(path)
    result = subprocess.run(
        ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def _lexical_absolute(path):
    return Path(os.path.abspath(str(Path(path).expanduser())))


def _lexical_repository_root(path):
    """Find the nearest containing worktree without following output symlinks."""
    lexical = _lexical_absolute(path)
    candidate = lexical.parent
    while candidate != candidate.parent:
        result = subprocess.run(
            ["git", "-C", str(candidate), "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            root = Path(result.stdout.strip()).resolve()
            try:
                lexical.relative_to(root)
                return root
            except ValueError:
                pass
        candidate = candidate.parent
    return None


def _reject_existing_symlink_components(path):
    lexical = _lexical_absolute(path)
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current = current / part
        try:
            value = os.lstat(str(current))
        except FileNotFoundError:
            break
        except OSError as error:
            raise RepositoryHygieneError(
                "cannot inspect generated-output path component: %s" % error)
        if stat.S_ISLNK(value.st_mode):
            raise RepositoryHygieneError(
                "generated-output path components must not be symbolic links")


def _directory_open_flags():
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _validate_open_directory(descriptor):
    value = os.fstat(descriptor)
    if not stat.S_ISDIR(value.st_mode):
        raise RepositoryHygieneError("generated-output path component is not a directory")
    if value.st_mode & 0o022 and not value.st_mode & stat.S_ISVTX:
        raise RepositoryHygieneError(
            "generated-output directories must not be group/world writable")


def _open_directory_tree(path, create):
    """Open an absolute directory by walking every component without symlinks."""
    lexical = _lexical_absolute(path)
    if not lexical.anchor:
        raise RepositoryHygieneError("generated-output directory must be absolute")
    flags = _directory_open_flags()
    descriptor = None
    try:
        descriptor = os.open(lexical.anchor, flags)
        _validate_open_directory(descriptor)
        for part in lexical.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o700, dir_fd=descriptor)
                child = os.open(part, flags, dir_fd=descriptor)
            try:
                _validate_open_directory(child)
            except Exception:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise RepositoryHygieneError(
            "cannot traverse generated-output directory safely: %s" % error)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        raise


def ensure_generated_path(path):
    """Reject generated output in a source-controlled location.

    Paths outside a Git worktree are allowed. Inside this worktree, generated
    output must live below a dedicated ignored root such as ``data/`` or
    ``downloads/`` and must match the active Git ignore rules.
    """
    resolved = _lexical_absolute(path)
    try:
        existing = os.lstat(str(resolved))
    except FileNotFoundError:
        existing = None
    except OSError as error:
        raise RepositoryHygieneError("cannot inspect generated output: %s" % error)
    if existing is not None and stat.S_ISLNK(existing.st_mode):
        raise RepositoryHygieneError("generated output must not be a symbolic link")
    _reject_existing_symlink_components(resolved)
    root = _lexical_repository_root(resolved)
    if root is None:
        return resolved
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return resolved
    if not relative.parts or relative.parts[0] not in GENERATED_ROOTS:
        raise RepositoryHygieneError(
            "generated output inside the repository must be below one of: %s" %
            ", ".join(sorted(GENERATED_ROOTS)))
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--",
         relative.as_posix()], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if tracked.returncode == 0:
        raise RepositoryHygieneError("refusing to overwrite a tracked path")
    ignored = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--quiet", "--no-index", "--",
         relative.as_posix()], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if ignored.returncode != 0:
        raise RepositoryHygieneError(
            "generated path is not covered by the repository ignore rules")
    return resolved


def open_generated_directory(path, create=True):
    """Securely create/open an ignored or external directory; caller closes fd."""
    if not isinstance(create, bool):
        raise RepositoryHygieneError("generated-directory create flag must be boolean")
    directory = ensure_generated_path(path)
    return directory, _open_directory_tree(directory, create=create)


def atomic_write_generated(path, content):
    """Atomically write bytes without following the destination entry."""
    if not isinstance(content, bytes):
        raise RepositoryHygieneError("generated content must be bytes")
    destination = ensure_generated_path(path)
    directory_fd = _open_directory_tree(destination.parent, create=True)
    temporary = None
    temporary_identity = None
    descriptor = None
    try:
        name = destination.name
        if name in ("", ".", "..") or os.sep in name or (os.altsep and os.altsep in name):
            raise RepositoryHygieneError("generated output filename is invalid")
        try:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and not stat.S_ISREG(current.st_mode):
            raise RepositoryHygieneError(
                "generated output must replace only a regular file")
        open_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            open_flags |= os.O_NOFOLLOW
        for unused_attempt in range(100):
            candidate = ".generated-%s.tmp" % secrets.token_hex(16)
            try:
                descriptor = os.open(candidate, open_flags, 0o600,
                                     dir_fd=directory_fd)
                temporary = candidate
                break
            except FileExistsError:
                continue
        if descriptor is None:
            raise RepositoryHygieneError("cannot allocate a unique generated temp file")
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise RepositoryHygieneError("generated temp file is not a private regular file")
        temporary_identity = (opened.st_dev, opened.st_ino)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        current_temp = os.stat(temporary, dir_fd=directory_fd, follow_symlinks=False)
        if ((current_temp.st_dev, current_temp.st_ino) != temporary_identity or
                not stat.S_ISREG(current_temp.st_mode) or current_temp.st_nlink != 1):
            raise RepositoryHygieneError("generated temp file changed before promotion")
        try:
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and not stat.S_ISREG(current.st_mode):
            raise RepositoryHygieneError(
                "generated output changed to a non-regular file before promotion")
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        temporary = None
        os.fsync(directory_fd)
    except OSError as error:
        raise RepositoryHygieneError("cannot write generated output: %s" % error)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if temporary is not None:
            try:
                current_temp = os.stat(
                    temporary, dir_fd=directory_fd, follow_symlinks=False)
                if ((current_temp.st_dev, current_temp.st_ino) == temporary_identity and
                        stat.S_ISREG(current_temp.st_mode) and current_temp.st_nlink == 1):
                    os.unlink(temporary, dir_fd=directory_fd)
            except OSError:
                pass
        os.close(directory_fd)
    return destination


def _tracked_paths(root):
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RepositoryHygieneError(
            "cannot inspect tracked files: %s" % result.stderr.decode("utf-8", "replace"))
    return tuple(value.decode("utf-8") for value in result.stdout.split(b"\0") if value)


def tracked_hygiene_violations(root):
    """Return tracked paths that look like generated data or build output."""
    root = Path(root).expanduser().resolve()
    violations = []
    for value in _tracked_paths(root):
        normalized = value.replace(os.sep, "/")
        if normalized in ALLOWED_TRACKED_FIXTURES:
            continue
        first = normalized.split("/", 1)[0]
        lowered = normalized.lower()
        components = normalized.split("/")
        lowered_components = lowered.split("/")
        if (first in FORBIDDEN_TRACKED_ROOTS or
                any(component in FORBIDDEN_TRACKED_COMPONENTS
                    or component.endswith(".egg-info")
                    for component in lowered_components) or
                lowered_components[-1] in FORBIDDEN_TRACKED_BASENAMES or
                lowered_components[-1].startswith(".coverage.") or
                any(lowered.endswith(suffix) for suffix in FORBIDDEN_TRACKED_SUFFIXES)):
            violations.append(normalized)
    return tuple(sorted(violations))


def assert_repository_hygiene(root):
    violations = tracked_hygiene_violations(root)
    if violations:
        raise RepositoryHygieneError(
            "tracked generated data/build outputs: %s" % ", ".join(violations))
    return violations
