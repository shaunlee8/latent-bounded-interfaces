# CUDA Support

The supported public CUDA code is limited to `cuda/interface/`.

That directory contains the backbone-agnostic suffix-scan extension used to compose bounded-interface pullback matrices during the LBI backward pass. Build it from the repository root with:

```bash
PYTHON_BIN=${PYTHON_BIN:-python} ./cuda/interface/build.sh
```

See `cuda/interface/README.md` for the extension API, expected tensor shapes, and test command.
