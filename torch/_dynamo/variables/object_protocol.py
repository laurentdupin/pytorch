"""
Implements CPython's type protocol for VariableTracker objects.

Following the TypeVariableTracker design doc, we mirror CPython's
PyTypeObject slots (e.g., tp_iter, tp_richcompare) by providing
generic dispatcher functions that route to per-type hook methods.

For example:
  - generic_getiter(tx, obj) corresponds to PyObject_GetIter(obj)
  - obj.iter_impl(tx) corresponds to type(obj)->tp_iter(obj)
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..symbolic_convert import InstructionTranslator
    from .base import VariableTracker


def generic_len(
    tx: "InstructionTranslator", obj: "VariableTracker"
) -> "VariableTracker":
    """
    Implements PyObject_Size/PyObject_Length semantics for VariableTracker objects.
    Routes to obj.len_impl(tx)
    """
    return obj.len_impl(tx)


# TODO(guilhermeleobas): should we narrow the return type to IteratorVariable?
def generic_getiter(
    tx: "InstructionTranslator", obj: "VariableTracker"
) -> "VariableTracker":
    """
    Implements PyObject_GetIter semantics for VariableTracker objects.
    Routes to obj.iter_impl(tx), the tp_iter slot on the object's type.
    """
    from .base import VariableTracker

    # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/abstract.c#2848
    # The algorithm for PyObject_GetIter is as follows: Steps:
    # 1. If the object has tp_iter slot, call it and return the result The
    #    return object must be an iterator (it must have a tp_iternext slot)
    # 2. If the object implements the sequence protocol - implements __getitem__
    #    and __len__, then create a sequence iterator for the object and return
    #    it.
    # 3. Otherwise, raise a TypeError
    if obj.__class__.iter_impl is not VariableTracker.iter_impl:
        return obj.iter_impl(tx)
    elif False and vt_sequence_check(obj):
        from .iter import SequenceIterator

        return SequenceIterator(obj)
    else:
        msg = VariableTracker.build(
            tx, f"'{obj.python_type_name()}' object is not iterable"
        )
        raise_observed_exception(
            TypeError,
            tx,
            args=[msg],
        )

    return obj.iter_impl(tx)
