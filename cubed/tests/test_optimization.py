import numpy as np
import pytest
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
from cubed.tests.utils import TaskCounter


@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, allowed_mem=100000)


def test_fusion(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.negative(a)
    c = xp.astype(b, np.float32)
    d = xp.negative(c)

    num_created_arrays = 4  # a, b, c, d
    assert d.plan.num_tasks(optimize_graph=False) == num_created_arrays + 12
    num_created_arrays = 2  # a, d
    assert d.plan.num_tasks(optimize_graph=True) == num_created_arrays + 4

    task_counter = TaskCounter()
    result = d.compute(callbacks=[task_counter])
    assert task_counter.value == num_created_arrays + 4

    assert_array_equal(
        result,
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32),
    )


def test_fusion_transpose(spec):
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)
    b = xp.negative(a)
    c = xp.astype(b, np.float32)
    d = c.T

    num_created_arrays = 4  # a, b, c, d
    assert d.plan.num_tasks(optimize_graph=False) == num_created_arrays + 12
    num_created_arrays = 2  # a, d
    assert d.plan.num_tasks(optimize_graph=True) == num_created_arrays + 4

    task_counter = TaskCounter()
    result = d.compute(callbacks=[task_counter])
    assert task_counter.value == num_created_arrays + 4

    assert_array_equal(
        result,
        np.array([[-1, -4, -7], [-2, -5, -8], [-3, -6, -9]]).astype(np.float32),
    )


def test_no_fusion(spec):
    # b can't be fused with c because d also depends on b
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.positive(a)
    c = xp.positive(b)
    d = xp.equal(b, c)

    num_created_arrays = 4  # a, b, c, d
    assert d.plan.num_tasks(optimize_graph=False) == num_created_arrays + 3
    assert d.plan.num_tasks(optimize_graph=True) == num_created_arrays + 3

    task_counter = TaskCounter()
    result = d.compute(callbacks=[task_counter])
    assert task_counter.value == num_created_arrays + 3

    assert_array_equal(result, np.ones((2, 2)))


def test_no_fusion_multiple_edges(spec):
    a = xp.ones((2, 2), chunks=(2, 2), spec=spec)
    b = xp.positive(a)
    c = xp.asarray(b)
    # b and c are the same array, so d has a single dependency
    # with multiple edges
    # this should not be fused under the current logic
    d = xp.equal(b, c)

    num_created_arrays = 3  # a, c, d
    assert d.plan.num_tasks(optimize_graph=False) == num_created_arrays + 2
    assert d.plan.num_tasks(optimize_graph=True) == num_created_arrays + 2

    task_counter = TaskCounter()
    result = d.compute(callbacks=[task_counter])
    assert task_counter.value == num_created_arrays + 2

    assert_array_equal(result, np.full((2, 2), True))