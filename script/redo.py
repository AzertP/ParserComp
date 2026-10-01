#!/usr/bin/env python3
"""
redo.py — minimal redo build system with integrated pipeline support.

Build:    redo.py <target>
Dep tracking: redo-ifchange dep...

Pipeline:
  - cmd.ls, cmd.grep, cmd['wc']
  - cmd.ls['-alt', '.']
  - cmd.ls | cmd.grep['foo'] | cmd.wc['-l']
  - cmd.ls | pycmd(fn) | cmd.wc['-l']
"""

import os, sys, shutil, subprocess, threading
from pathlib import Path
from subprocess import PIPE

__all__ = ["cmd", "pycmd", "PyCommand", "redo_ifchange"]

# ── Pipeline ──────────────────────────────────────────────────────────────────

class CommandNotFound(Exception): pass

class BaseCommand:
    def __or__(self, other): return Pipeline(self, other)
    def __getitem__(self, args):
        if not isinstance(args, (tuple, list)): args = (args,)
        return self._bind(*args)
    def _bind(self, *args): raise NotImplementedError
    def __call__(self): return self.run()[1]
    def run(self, check=False, stream=False):
        kwargs = {} if stream else {"stdout": PIPE, "stderr": PIPE}
        proc = self.popen(**kwargs)
        stdout, stderr = (None, None) if stream else proc.communicate()
        rc = proc.wait() if stream else proc.returncode
        if check and rc != 0: raise subprocess.CalledProcessError(rc, repr(self))
        return (rc,
            stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout,
            stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr)
    def popen(self, **kwargs): raise NotImplementedError
    def stderr_to_stdout(self):
        return _RedirectedCommand(self)

class _RedirectedCommand(BaseCommand):
    def __init__(self, inner): self.inner = inner
    def _bind(self, *args): return _RedirectedCommand(self.inner._bind(*args))
    def popen(self, **kwargs):
        kwargs["stderr"] = subprocess.STDOUT
        return self.inner.popen(**kwargs)
    def __repr__(self): return f"{self.inner!r} 2>&1"

class ConcreteCommand(BaseCommand):
    def __init__(self, path, args=(), env=None):
        self._path, self._args, self._env = path, tuple(str(a) for a in args), env
    def _bind(self, *args):
        return ConcreteCommand(self._path, self._args + tuple(str(a) for a in args), self._env)
    def with_env(self, env): return ConcreteCommand(self._path, self._args, env)
    def popen(self, **kwargs):
        if self._env: kwargs.setdefault("env", self._env)
        return subprocess.Popen([self._path, *self._args], **kwargs)
    def __repr__(self): return " ".join([os.path.basename(self._path), *self._args])

class Pipeline(BaseCommand):
    def __init__(self, src, dst): self.src, self.dst = src, dst
    def popen(self, **kwargs):
        srcproc = self.src.popen(**{**kwargs, "stdout": PIPE})
        dstproc = self.dst.popen(**{**kwargs, "stdin": srcproc.stdout})
        if srcproc.stdout: srcproc.stdout.close()
        _wrap_wait(dstproc, srcproc)
        _wrap_communicate(dstproc, srcproc)
        return dstproc
    def __repr__(self): return f"{self.src!r} | {self.dst!r}"

def _wrap_wait(dstproc, srcproc):
    orig = dstproc.wait
    def wait(*a, **kw): rc = orig(*a, **kw); srcproc.wait(); return rc
    dstproc.wait = wait

def _wrap_communicate(dstproc, srcproc):
    orig = dstproc.communicate
    def communicate(input=None): out, err = orig(input); srcproc.wait(); return out, err
    dstproc.communicate = communicate

class PyProc:
    returncode = None; stderr = None; stdin = None
    def __init__(self, fn, in_fd, out_mode):
        self._in_fd = os.fdopen(os.dup(in_fd.fileno()), "rb") if in_fd else None
        self.stdout = self._out_w = None
        self._write_to_parent_stdout = False
        if out_mode == PIPE:
            r, w = os.pipe()
            self.stdout, self._out_w = os.fdopen(r, "rb"), os.fdopen(w, "wb")
        elif out_mode is None:
            self._write_to_parent_stdout = True
        self._thread = threading.Thread(target=self._run, args=(fn,), daemon=True)
        self._thread.start()
    def _emit(self, data: bytes):
        if self._out_w: self._out_w.write(data); self._out_w.flush()
        elif self._write_to_parent_stdout: sys.stdout.buffer.write(data); sys.stdout.buffer.flush()
    def _run(self, fn):
        try:
            if self._in_fd is None: self.returncode = 0; return
            for line in self._in_fd:
                result = fn(line)
                if result is None: continue
                self._emit(result.encode() if isinstance(result, str) else result)
            self.returncode = 0
        except Exception as e:
            sys.stderr.buffer.write(f"PyCommand error: {e}\n".encode())
            self.returncode = 1
        finally:
            try:
                if self._in_fd: self._in_fd.close()
            except Exception: pass
            if self._out_w: self._out_w.close()
    def wait(self, *a, **kw): self._thread.join(); return self.returncode
    def communicate(self, input=None): return (self.stdout.read() if self.stdout else b""), b""
    def poll(self): return None if self._thread.is_alive() else self.returncode

class PyCommand(BaseCommand):
    def __init__(self, fn): self.fn = fn
    def popen(self, **kwargs): return PyProc(self.fn, kwargs.get("stdin"), kwargs.get("stdout"))
    def __repr__(self): return f"<py:{self.fn.__name__}>"

pycmd = PyCommand

class _CmdNamespace:
    def __getattr__(self, name):
        path = shutil.which(name)
        if path is None: raise CommandNotFound(f"Command not found: {name!r}")
        command = ConcreteCommand(path)
        setattr(self, name, command); return command
    def __getitem__(self, name): return getattr(self, name)

cmd = _CmdNamespace()

# ── Redo ──────────────────────────────────────────────────────────────────────

REDO_DIR = Path(".redo")
REDO_DIR.mkdir(exist_ok=True)

def depfile(t):  return REDO_DIR / f"{t}.deps"
def tmpdeps(t): return REDO_DIR / f"{t}.deps.tmp"
def tmpout(t):  return REDO_DIR / f"{t}.out.tmp"

def find_dofile(t: Path) -> Path | None:
    if (p := Path(f"{t}.do.py")).exists(): return p
    for d in [t.parent, Path(".")]:
        if (p := d / f"default{t.suffix}.do.py").exists(): return p

def is_outdated(t: Path) -> bool:
    if not t.exists(): return True
    if (do := find_dofile(t)) and do.stat().st_mtime > t.stat().st_mtime: return True
    df = depfile(t)
    return df.exists() and any(
        not (d := Path(l)).exists() or d.stat().st_mtime > t.stat().st_mtime
        for l in df.read_text().splitlines()
    )

def build(target: str) -> None:
    t, stack = Path(target), os.environ.get("REDO_STACK", "")
    if target in stack.split(":"):
        print(f"redo: cycle at {target} (stack: {stack})", file=sys.stderr); sys.exit(1)
    if not is_outdated(t): return
    if not (do := find_dofile(t)):
        print(f"redo: no rule to build {target}", file=sys.stderr); sys.exit(1)
    td, to = tmpdeps(t), tmpout(t)
    td.parent.mkdir(parents=True, exist_ok=True)
    td.write_text("")
    if to.exists(): to.unlink()
    env = {**os.environ,
        "REDO_STACK":    f"{stack}:{target}" if stack else target,
        "REDO_TARGET":   target,
        "REDO_DEPS_TMP": str(td),
        "REDO_PYTHON":   sys.executable,
        "REDO_SCRIPT":   str(Path(__file__).resolve()),
        "PYTHONPATH":    str(Path(__file__).resolve().parent.parent)
                         + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
    }
    try:
        cmd[sys.executable].with_env(env)[str(do), t.name, target, str(to)].run(check=True, stream=True)
        if to.exists(): os.replace(to, t)
        os.replace(td, depfile(t))
    except Exception:
        for f in [to, td]:
            try:
                if f.exists(): f.unlink()
            except OSError:
                try: f.open("w").close()  # truncate as fallback (macOS-mounted fs)
                except OSError: pass
        sys.exit(1)

def redo_ifchange(*deps: str) -> None:
    if not (td := os.environ.get("REDO_DEPS_TMP")):
        print("redo_ifchange: not inside a .do.py script", file=sys.stderr); sys.exit(1)
    with open(td, "a") as f:
        for dep in deps:
            # only build deps that have a rule; static source files are just tracked
            if find_dofile(Path(dep)):
                build(dep)
            f.write(dep + "\n")

def dofile_init():
    target = Path(sys.argv[2])
    tmpout = Path(sys.argv[3])
    return target, tmpout

def main() -> None:
    if Path(sys.argv[0]).name == "redo-ifchange":
        if len(sys.argv) < 2: print("usage: redo-ifchange dep...", file=sys.stderr); sys.exit(2)
        redo_ifchange(*sys.argv[1:]); return
    if len(sys.argv) != 2: print("usage: redo target", file=sys.stderr); sys.exit(2)
    build(sys.argv[1])

if __name__ == "__main__": main()
