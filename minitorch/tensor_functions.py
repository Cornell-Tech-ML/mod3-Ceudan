"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

# import debugpy

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Returns the negative of the input tensor"""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Returns the gradient of the loss with respect to the input"""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Returns the inverse of the input tensor"""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Returns the gradient of the loss with respect to the input"""
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Returns the sum of two tensors"""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns the gradient of the loss with respect to the input"""
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all are true"""
        # val = int(dim.item())
        val = int(dim[0])
        # if dim is not None:
        #     return a.f.mul_reduce(a, int(dim.item()))\
        if val != -77:
            return a.f.mul_reduce(a, val)
        else:
            """REMEMBER THIS LINE: MIGHT REPLICATE IN TENSOR_ADD FUNCTION"""
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# TODO: Implement for Task 2.3.
class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Returns the product of two tensors"""
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, ...]:
        """Returns the gradient of the loss with respect to the input"""
        (t1, t2) = ctx.saved_values
        # t1   (5,1,4)
        # t2     (3,1)
        # prod (5,3,4)

        # # I am going to assume that the dimensions of t1 and t2 are the same, because I cannot yet add new dimensions
        # grad1 = grad_output.f.mul_zip(t2,grad_output)
        # for i in range(grad1.dims):
        #     neg_ind = -(i+1)
        #     if((i>=t1.dims) or (t1.shape[neg_ind]==1)):
        #         # if the size of this dimension was 1 or did not exist in original tensor (i.e. it was added in multiplication broadcasting) sum it out via summation of multiple gradient paths theorom
        #         grad1 = grad1.f.add_reduce(grad1,i)
        # # view it back to original shape
        # grad1 = non_differentiable_view(grad1, t1.shape)

        # grad2 = grad_output.f.mul_zip(t1,grad_output)
        # for i in range(grad2.dims):
        #     neg_ind = -(i+1)
        #     if((i>=t2.dims) or (t2.shape[neg_ind]==1)):
        #         grad2 = grad2.f.add_reduce(grad2,i)
        # grad2 = non_differentiable_view(grad2, t2.shape)

        # return grad_output.f.inv_back_zip(t1, grad_output)
        return grad_output.f.mul_zip(grad_output, t2), grad_output.f.mul_zip(
            grad_output, t1
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Returns the sigmoid of the input tensor"""
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Returns the gradient of the loss with respect to the input"""
        (out,) = ctx.saved_values
        negout = grad_output.f.neg_map(out)
        one_minus_out = grad_output.f.add_zip(grad_output._ensure_tensor(1), negout)
        grad = grad_output.f.mul_zip(out, one_minus_out)
        grad = grad_output.f.mul_zip(grad, grad_output)
        return grad


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Returns the ReLU of the input tensor"""
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Returns the gradient of the loss with respect to the input"""
        (t1,) = ctx.saved_values
        # grad = t1*0 + 1
        grad = grad_output.f.relu_back_zip(t1, grad_output)
        # filt = t1>0
        # grad = grad*filt
        # return grad_output.f.inv_back_zip(t1, grad_output)
        return grad


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Returns the log of the input tensor"""
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Returns the gradient of the loss with respect to the input"""
        (t1,) = ctx.saved_values
        grad = grad_output.f.log_back_zip(t1, grad_output)
        return grad


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Returns the exponential of the input tensor"""
        # print("Exp input:",t1)
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        # print("Exp output:",out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Returns the gradient of the loss with respect to the input"""
        (out,) = ctx.saved_values
        # print("Exp input",t1)
        grad = grad_output.f.mul_zip(out, grad_output)
        # print("Exp input grad:",grad_output)
        # print("Exp output grad:",grad)
        return grad


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, dim: Tensor) -> Tensor:
        """Sums the tensor along the specified dimension"""
        # """Needs fixing"""
        ctx.save_for_backward(t1, dim)
        val = int(dim[0])
        # if(val == -77):
        #     # if dims not specified, sum all dims
        #     for i in range(t1._tensor.dims):
        #         t1 = t1.f.add_reduce(t1,i)
        #     # reduce the dimension to 1
        #     # t1 = non_differentiable_view(t1,(1,))
        #     return t1
        # else:
        #     # sum only the specified dim
        #     return t1.f.add_reduce(t1,val)\
        return t1.f.add_reduce(t1, val)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, ...]:
        """Returns gradients of loss to inputs"""
        (t1, dim) = ctx.saved_values
        # grad = non_differentiable_view(grad_output, t1.shape)
        # return grad, grad_output._ensure_tensor(0)
        return grad_output, grad_output._ensure_tensor(0)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Check if t2 is less than t1"""
        ctx.save_for_backward(t1, t2)
        return t1.f.lt_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, ...]:
        """Returns zeros for the gradient"""
        (t1, t2) = ctx.saved_values
        # return grad_output.f.inv_back_zip(t1, grad_output)
        return t1.zeros(), t2.zeros()


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Check if two tensors are equal"""
        ctx.save_for_backward(t1, t2)
        return t1.f.eq_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, ...]:
        """Returns zeros for the gradient"""
        (t1, t2) = ctx.saved_values
        return t1.zeros(), t2.zeros()


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Check if two tensors are close"""
        ctx.save_for_backward(t1, t2)
        return t1.f.is_close_zip(t1, t2)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, order: Tensor) -> Tensor:
        """Permute the tensor to a new order"""
        # ctx.save_for_backward(order)
        ctx.save_for_backward(t1.shape)
        order_int = [int(val) for val in order.to_numpy()]
        t1._tensor = t1._tensor.permute(*order_int)
        return t1

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Needs fixing"""
        (orig_shape,) = ctx.saved_values
        grad = minitorch.Tensor.make(
            grad_output._tensor._storage, orig_shape, backend=grad_output.backend
        )
        # print("Permute order:",order)
        # inv_order = [0 for i in range(len(order))]
        # for i in range(len(order)):
        #     inv_order[order[i]] = i
        # grad_output._tensor = grad_output._tensor.permute(*inv_order)
        # order_der = tuple([0 for i in range(len(order))])
        # order_der = grad_output._ensure_tensor(order_der)
        return grad, 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """View the tensor to a new shape"""
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        # try:
        #     shape2 = [int(shape[i]) for i in range(shape.size)]
        # except Exception as e:
        #     # Print the error message
        #     print(f"ERROR IN VIEW FORWARD: {e}")
        #     print("shape",shape._tensor._storage)
        #     print("type shape",type(shape))
        #     print("size shape",shape._tensor.shape)
        #     print("size shape",shape.size)
        #     for i in range(shape.size):
        #         print("shape val type:",type(shape[i]))
        #         print("shape val:",shape[i])
        #     debugpy.breakpoint()
        # try:
        #     shape2 = [int(shape[i]) for i in range(shape.size)]
        # except:
        #     print("ERROR IN VIEW FORWARD \nshape-",shape._tensor,"\nshape size-",shape.size,"\na shape-",a.shape)
        # print("shape:",shape)
        # print("shape size:",shape.size)
        # print("a shape:",a.shape)
        # shape2 = [int(shape[i]) for i in range(len(shape._tensor.shape))]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )
        """original code below (I removed the tuple applied on shape which should already be a tuple)"""
        # return minitorch.Tensor.make(
        #     a._tensor._storage, tuple(shape2), backend=a.backend
        # )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Matrix Multiply backward (module 3)"""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the central difference for a function."""
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        try:
            np.testing.assert_allclose(
                x.grad[ind],
                check,
                1e-2,
                1e-2,
                err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
            )
        except Exception as e:
            print("ERROR IN GRAD CHECK")
            print("ERROR MESSAGE:", e)
            print(f"my grad x[{i}]:", x.grad[ind])
            ("their grad:", check)
        #  debugpy.breakpoint()

        # print("x.grad:",x.grad)
        # print("x.grad[ind]:",x.grad[ind])
        # print("check:",check)


# def non_differentiable_view(tens: Tensor, shape: Tuple[int]) -> Tensor:
#     """Reimpliment the view function but without the derivatives"""
#     assert tens._tensor.is_contiguous(), "Must be contiguous to view"
#     # shape = tens.shape
#     try:
#         shape2 = [int(shape[i]) for i in range(len(shape))]
#     except:
#         print("ERROR WITH SHAPES CUSTOM VIEW")
#         print("t1 shape:", shape)
#         print("t1 shape size:", shape.size)
#         print("grad shape:", tens.shape)
#         # raise BaseException("ERROR WITH SHAPES IN SUM BACKWARD")
#     # shape2 = [int(shape[i]) for i in range(len(shape._tensor.shape))]
#     # print("tuple shape2:",tuple(shape2))
#     tens = minitorch.Tensor.make(
#         tens._tensor._storage, tuple(shape2), backend=tens.backend
#     )
#     return tens
