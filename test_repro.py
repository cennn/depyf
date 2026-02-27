"""Minimal reproduction of BUILD_LIST decompilation error.

The bug: when torch.compile generates bytecode where both if/else branches
end with RETURN_VALUE, the decompiler's generic_jump_if computes an empty
else-body range (end_index == jump_index). The else-body then gets decompiled
as sequential code with the wrong stack state, causing BUILD_LIST to encounter
a Python None value (from LOAD_GLOBAL's NULL marker) and crash with:
    TypeError: sequence item 1: expected str instance, NoneType found
"""
import torch
import depyf

@torch.compile
def minimal_repro(a, b):
    if b.sum() < 0:
        b = b * -1
    return a * b

def main():
    with depyf.prepare_debug("./dump_repro_dir"):
        for _ in range(100):
            minimal_repro(torch.randn(10), torch.randn(10))

if __name__ == "__main__":
    main()
