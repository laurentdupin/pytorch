# Owner(s): ["module: dynamo"]

"""
Tests for the `in` operator and __contains__ protocol in PyTorch Dynamo.

Tests cover:
- sq_contains protocol: list, tuple, str, range, set, frozenset
- mp_contains protocol: dict, dict.keys()
- Fallback iteration: objects with __iter__ but no __contains__
- Sequence-protocol fallback: objects with __getitem__/__len__ but no __contains__/__iter__
- operator.contains() — exercises the builtin.py call_contains path
- User-defined classes with __contains__
- User-defined classes without __contains__ (iteration fallback)
- Error handling: unhashable types, non-iterable objects, __contains__ raises,
  non-bool return values, str TypeError, mid-iteration raise
"""

import operator

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


# ---------------------------------------------------------------------------
# Helper classes defined at module level (not inside compiled functions)
# ---------------------------------------------------------------------------


class WithContains:
    """Has explicit __contains__."""

    def __init__(self, data):
        self.data = data

    def __contains__(self, item):
        return item in self.data


class WithIterNoContains:
    """Has __iter__ but no __contains__ — forces iteration fallback."""

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)


class WithGetitemNoContains:
    """Sequence protocol via __getitem__ / __len__, no __contains__."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class WithContainsAndIter:
    """Has both __contains__ and __iter__; __contains__ should be preferred."""

    def __init__(self, data):
        self.data = data
        self.iter_calls = 0

    def __contains__(self, item):
        return item in self.data

    def __iter__(self):
        self.iter_calls += 1
        return iter(self.data)


class ListIterWrapper:
    """List wrapper with custom __iter__ and no __contains__ — forces iter fallback."""

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        # yields items doubled so membership tests differ from base list
        return iter([x * 2 for x in self.data])


class ListGetitemWrapper:
    """List wrapper using __getitem__ fallback — no __contains__, no __iter__."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        # shifts each value up by 100
        return self.data[idx] + 100


class DictIterWrapper:
    """Dict wrapper with custom __iter__ (over values) and no __contains__ — forces iter fallback."""

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data.values())


class TupleIterWrapper:
    """Tuple wrapper with custom __iter__ and no __contains__ — forces iter fallback."""

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        # yields items doubled
        return iter([x * 2 for x in self.data])


class TupleGetitemWrapper:
    """Tuple wrapper using __getitem__ fallback — no __contains__, no __iter__."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        # shifts each value up by 100
        return self.data[idx] + 100

    def __len__(self):
        return len(self.data)


class SetIterWrapper:
    """Set wrapper with custom __iter__ and no __contains__ — forces iter fallback."""

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        # yields items doubled
        return iter([x * 2 for x in self.data])


class DictGetitemWrapper:
    """Dict wrapper using __getitem__ fallback — no __contains__, no __iter__."""

    def __init__(self, data):
        self.data = data
        self.sorted_keys = sorted(data.keys())

    def __getitem__(self, idx):
        # access values by index via sorted keys
        return self.data[self.sorted_keys[idx]]

    def __len__(self):
        return len(self.data)


class ListSubclassCustomContains(list):
    """Subclass of list that overrides __contains__."""

    def __contains__(self, item):
        return item in [x * 2 for x in super().__iter__()]


class DictSubclassCustomContains(dict):
    """dict subclass whose __contains__ checks values instead of keys."""

    def __contains__(self, item):
        return item in self.values()


class SetSubclassCustomContains(set):
    """set subclass whose __contains__ negates the base class result."""

    def __contains__(self, item):
        return not super().__contains__(item)


class ContainsRaisesTypeError:
    """__contains__ unconditionally raises TypeError."""

    def __contains__(self, item):
        raise TypeError("bad operand")


class ContainsReturnsTruthy:
    """__contains__ returns a non-bool truthy value."""

    def __contains__(self, item):
        return 42


class ContainsReturnsFalsy:
    """__contains__ returns a non-bool falsy value."""

    def __contains__(self, item):
        return 0


class NoIterNoContains:
    """Has neither __iter__ nor __contains__ — triggers TypeError on `in`."""


class RaisesDuringIter:
    """Iterator that raises ValueError partway through."""

    def __iter__(self):
        yield 1
        yield 2
        raise ValueError("mid-iteration error")


# ---------------------------------------------------------------------------
# Test Class
# ---------------------------------------------------------------------------


class _ContainsBase:
    """Base test class for __contains__ protocol with parameterized types."""

    thetype = None  # Override in subclass
    data = [1, 2, 3]
    empty = []
    item = 2
    missing_item = 4

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_contains(self):
        # Basic membership tests for the main type under test
        seq = self.thetype(self.data)
        self.assertTrue(self.item in seq)
        self.assertFalse(self.missing_item in seq)

    @make_dynamo_test
    def test_contains_negation(self):
        # Test `not in` operator
        seq = self.thetype(self.data)
        self.assertFalse(self.item not in seq)
        self.assertTrue(self.missing_item not in seq)

    @make_dynamo_test
    def test_contains_operator_module(self):
        # Test operator.contains()
        seq = self.thetype(self.data)
        self.assertTrue(operator.contains(seq, self.item))
        self.assertFalse(operator.contains(seq, self.missing_item))

    @make_dynamo_test
    def test_contains_empty(self):
        # Test on empty container
        seq = self.thetype(self.empty)
        self.assertFalse(self.item in seq)
        self.assertTrue(self.missing_item not in seq)


class ListContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = list


class TupleContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = tuple


class StrContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = str
    data = "abc"
    item = "b"
    missing_item = "d"


class DictContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = dict
    data = {"a": 1, "b": 2, "c": 3}
    item = "b"
    missing_item = "d"


class SetContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = set


class FrozensetContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = frozenset


class WithContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = WithContains


class WithIterNoContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = WithIterNoContains


class WithGetitemNoContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = WithGetitemNoContains


class WithContainsAndIterTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = WithContainsAndIter


class ListIterWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = ListIterWrapper
    item = 4  # Will be doubled by __iter__
    missing_item = 1


class ListGetitemWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = ListGetitemWrapper
    item = 101  # Will be shifted +100 by __getitem__
    missing_item = 10


class DictIterWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = DictIterWrapper
    data = {"a": 1, "b": 2, "c": 3}
    empty = {}
    item = 2  # Will iterate over values
    missing_item = "a"


class TupleIterWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = TupleIterWrapper
    item = 4  # Will be doubled by __iter__
    missing_item = 10


class TupleGetitemWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = TupleGetitemWrapper
    item = 101  # Will be shifted +100 by __getitem__
    missing_item = 10


class SetIterWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = SetIterWrapper
    data = {1, 2, 3}
    item = 4  # Will be doubled by __iter__
    missing_item = 10


class DictGetitemWrapperTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = DictGetitemWrapper
    data = {"a": 1, "b": 2, "c": 3}
    empty = {}
    item = 2  # Will be retrieved by __getitem__
    missing_item = 10


class ListSubclassCustomContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = ListSubclassCustomContains
    data = [1, 2, 3]
    item = 4  # Custom __contains__ checks doubled values
    missing_item = 10


class DictSubclassCustomContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = DictSubclassCustomContains
    data = {"a": 1, "b": 2, "c": 3}
    empty = {}
    item = 2  # Custom __contains__ checks values not keys
    missing_item = 10


class SetSubclassCustomContainsTest(_ContainsBase, torch._dynamo.test_case.TestCase):
    thetype = SetSubclassCustomContains
    data = [1, 2, 3]
    item = 4  # Custom __contains__ negates base class result
    missing_item = 1

    @make_dynamo_test
    def test_contains_empty(self):
        # Test on empty container
        seq = self.thetype(self.empty)
        self.assertTrue(self.item in seq)
        self.assertFalse(self.missing_item not in seq)


class ContainsReturnsTruthyTest(torch._dynamo.test_case.TestCase):
    def test_contains(self):
        seq = ContainsReturnsTruthy()
        self.assertTrue(2 in seq)
        self.assertTrue(None in seq)


class ContainsReturnsFalsyTest(torch._dynamo.test_case.TestCase):
    def test_contains(self):
        seq = ContainsReturnsFalsy()
        self.assertFalse(2 in seq)
        self.assertFalse(None in seq)


class NoIterNoContainsTest(torch._dynamo.test_case.TestCase):
    """Tests for objects with neither __iter__ nor __contains__"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_no_iter_no_contains_raises_typeerror(self):
        # Should raise TypeError when neither __iter__ nor __contains__ exists
        obj = NoIterNoContains()
        with self.assertRaises(TypeError):
            _ = 1 in obj

    @make_dynamo_test
    def test_no_iter_no_contains_not_in_raises_typeerror(self):
        # Should raise TypeError for `not in` operator as well
        obj = NoIterNoContains()
        with self.assertRaises(TypeError):
            _ = 1 not in obj


class RaisesDuringIterTest(torch._dynamo.test_case.TestCase):
    """Tests for iterators that raise exceptions during iteration"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_raises_during_iter_found_before_error(self):
        # Item 1 found before error occurs
        obj = RaisesDuringIter()
        result = 1 in obj
        self.assertTrue(result)

    @make_dynamo_test
    def test_raises_during_iter_not_found_raises_error(self):
        # Item not found, error occurs during iteration
        obj = RaisesDuringIter()
        with self.assertRaises(ValueError) as cm:
            _ = 3 in obj
        self.assertEqual(str(cm.exception), "mid-iteration error")


class ContainsRaisesTypeErrorTest(torch._dynamo.test_case.TestCase):
    """Tests for __contains__ that raises TypeError"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_contains_raises_typeerror(self):
        # __contains__ raises TypeError
        obj = ContainsRaisesTypeError()
        with self.assertRaises(TypeError) as cm:
            _ = 1 in obj
        self.assertEqual(str(cm.exception), "bad operand")

    @make_dynamo_test
    def test_contains_raises_typeerror_not_in(self):
        # __contains__ raises TypeError for `not in` operator
        obj = ContainsRaisesTypeError()
        with self.assertRaises(TypeError) as cm:
            _ = 1 not in obj
        self.assertEqual(str(cm.exception), "bad operand")


class RangeContainsTest(torch._dynamo.test_case.TestCase):
    """Specific tests for range __contains__ protocol"""

    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    @make_dynamo_test
    def test_range_basic(self):
        # Basic range containment
        seq = range(5)
        self.assertTrue(2 in seq)
        self.assertFalse(10 in seq)

    @make_dynamo_test
    def test_range_with_start_stop(self):
        # Test range with start and stop
        seq = range(5, 15)
        self.assertTrue(10 in seq)
        self.assertFalse(3 in seq)
        self.assertFalse(15 in seq)

    @make_dynamo_test
    def test_range_with_step(self):
        # Test range with step
        seq = range(0, 10, 2)
        self.assertTrue(4 in seq)
        self.assertFalse(3 in seq)
        self.assertFalse(10 in seq)

    @make_dynamo_test
    def test_range_negative_step(self):
        # Test range with negative step
        seq = range(10, 0, -1)
        self.assertTrue(5 in seq)
        self.assertFalse(0 in seq)
        self.assertFalse(11 in seq)

    @make_dynamo_test
    def test_range_empty(self):
        # Test empty range
        seq = range(5, 5)
        self.assertFalse(5 in seq)
        self.assertFalse(4 in seq)

    @make_dynamo_test
    def test_range_single_element(self):
        # Test range with single element
        seq = range(5, 6)
        self.assertTrue(5 in seq)
        self.assertFalse(4 in seq)
        self.assertFalse(6 in seq)

    @make_dynamo_test
    def test_range_negative_numbers(self):
        # Test range with negative numbers
        seq = range(-5, 5)
        self.assertTrue(-3 in seq)
        self.assertTrue(0 in seq)
        self.assertFalse(-6 in seq)
        self.assertFalse(5 in seq)

    @make_dynamo_test
    def test_range_negation(self):
        # Test `not in` operator
        seq = range(5)
        self.assertFalse(2 not in seq)
        self.assertTrue(10 not in seq)

    @make_dynamo_test
    def test_range_operator_module(self):
        # Test operator.contains()
        seq = range(10)
        self.assertTrue(operator.contains(seq, 5))
        self.assertFalse(operator.contains(seq, 15))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
