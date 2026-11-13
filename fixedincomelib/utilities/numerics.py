import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """Smooth Hagan-West style monotone convex IFR interpolator (node-interpolating).

    Brief: continuous, passes through the supplied IFR points, monotone and convex within each node
    interval, FLAT extrapolation, constant for one node.

    Ideas taken from Hagan & West (Wilmott paper, Section 6):
      * local construction: only the neighbouring intervals influence a piece;
      * node values are derived from neighbours by the length-weighted average of equation (22);
      * a plain quadratic is not always monotone, so the interval is split in two pieces at a point chosen
        "at exactly the right moment" so that a constraint (there: integral of g = 0) holds exactly
        (equations (28)-(31)).
    Here the same device is applied to the SLOPE, because the curve must pass through the node values.

    One interval [t_i, t_{i+1}], h = width, y0, y1 = node IFRs, d = (y1 - y0) / h = secant slope.
      * node slopes m_i: HW weights (22) applied to the two neighbouring secants,
            m_i = (h_i d_{i-1} + h_{i-1} d_i) / (h_{i-1} + h_i)  if d_{i-1}, d_i have the same sign, else 0;
        ends: slope 0 where flat extrapolation can continue smoothly (left end for rising data, right end
        for falling data), as the paper's end values (23), (24) are chosen so that f' = 0 at the ends;
        otherwise m_0 = 2 d_0 - m_1, m_n = 2 d_{n-1} - m_{n-1}, so the two end slopes average to the secant.
      * interval end slopes s0 (left), s1 (right) are the node slopes clipped so that
            s0 <= d <= s1  (slope only rises: convex)  and  s0, s1 of the sign of d  (never changes sign: monotone):
            d >= 0:  s0 = min(max(m_i, 0), d),  s1 = max(m_{i+1}, d)
            d <  0:  s0 = min(m_i, d),          s1 = min(max(m_{i+1}, d), 0)
      * slope goes linearly s0 -> d on [0, k] and d -> s1 on [k, h]. The mean slope is d, so the curve
        ends exactly at y1, iff  k = h (s1 - d) / (s1 - s0). The curve is two parabolas:
            f = y0 + s0 u + a u^2 / (2k),                         0 <= u <= k,   a = d - s0
            f = y_k + d u + b u^2 / (2 (h - k)),                  0 <= u <= h-k, b = s1 - d
    Continuous, hits both nodes, monotone, convex (f'' = a/k or b/(h-k) >= 0). For convex data (secants
    rising) the clipping does nothing and the slopes agree on both sides of every node, so the curve is
    smooth (C1). Where the data is not convex at a node, no curve can be convex on both sides and smooth
    there; the slopes are clipped, the curve keeps all the properties above and has a kink at that node.
    Valid for any sign of the data. Monotone between the nodes, so positive data give a positive curve.
    The integral is closed form (every piece is a quadratic).

    Extrapolation is FLAT. With one node the curve is constant.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        if self.extrap_method_ != ExtrapMethod.FLAT:
            raise ValueError("MONOTONE_CONVEX supports FLAT extrapolation only.")
        if self.length_ == 0:
            raise ValueError("At least one node is required.")
        x = np.asarray(self.axis1_, dtype=float)
        y = np.asarray(self.values_, dtype=float)
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("Nodes and values must be finite.")

        # duplicate node times are dropped (first value kept, as in the PCP)
        x, idx = np.unique(x, return_index=True)
        y = y[idx]
        self.x_, self.y_ = x, y
        n = len(x)

        # One quadratic piece: f(t) = c0 + c1 (t - start) + c2 (t - start)^2 on [start, end]
        starts, ends, coefs = [], [], []
        if n >= 2:
            h = np.diff(x)
            d = np.diff(y) / h
            # node slopes: HW weights (22) on the secants; 0 where the data turns
            m = np.zeros(n)
            for i in range(1, n - 1):
                if d[i - 1] * d[i] > 0:
                    m[i] = (h[i] * d[i - 1] + h[i - 1] * d[i]) / (h[i - 1] + h[i])
            # end slopes: slope 0 where the flat extrapolation can continue smoothly (rising data at the
            # left end, falling data at the right end; the paper's (23)/(24) also give f' = 0 at the ends);
            # otherwise m_0 = 2 d_0 - m_1, so the two end slopes average to the secant.
            if n == 2:
                m[0], m[1] = (0.0, 2.0 * d[0]) if d[0] >= 0 else (2.0 * d[0], 0.0)
            else:
                m[0] = 0.0 if d[0] > 0 else 2.0 * d[0] - m[1]
                m[-1] = 0.0 if d[-1] < 0 else 2.0 * d[-1] - m[-2]
            for i in range(n - 1):
                if d[i] >= 0:
                    s0, s1 = min(max(m[i], 0.0), d[i]), max(m[i + 1], d[i])
                else:
                    s0, s1 = min(m[i], d[i]), min(max(m[i + 1], d[i]), 0.0)
                a, b = d[i] - s0, s1 - d[i]
                k = h[i] * b / (a + b) if a + b > 0 else h[i]   # break point that makes the mean slope d
                if k > 0:
                    starts.append(x[i]); ends.append(x[i] + k)
                    coefs.append((y[i], s0, a / (2 * k)))
                if k < h[i]:
                    y_k = y[i] + k * (s0 + d[i]) / 2
                    starts.append(x[i] + k); ends.append(x[i + 1])
                    coefs.append((y_k, d[i], b / (2 * (h[i] - k))))
        self.starts_ = np.array(starts)
        self.ends_ = np.array(ends)
        self.coefs_ = np.array(coefs).reshape(-1, 3)

    def interpolate(self, x: float) -> float:
        if x <= self.x_[0]:
            return float(self.y_[0])
        if x >= self.x_[-1]:
            return float(self.y_[-1])
        k = int(np.searchsorted(self.starts_, x, side='right')) - 1
        c0, c1, c2 = self.coefs_[k]
        u = x - self.starts_[k]
        return float(c0 + c1 * u + c2 * u * u)

    def integrate(self, start_x: float, end_x: float) -> float:
        if end_x < start_x:
            return -self.integrate(end_x, start_x)
        first, last = float(self.x_[0]), float(self.x_[-1])
        total = self.y_[0] * max(min(end_x, first) - start_x, 0.0)        # flat to the left
        total += self.y_[-1] * max(end_x - max(start_x, last), 0.0)      # flat to the right
        if len(self.starts_) > 0:
            lo = np.clip(start_x, self.starts_, self.ends_) - self.starts_
            hi = np.clip(end_x, self.starts_, self.ends_) - self.starts_
            c0, c1, c2 = self.coefs_.T
            total += np.sum(c0 * (hi - lo) + c1 * (hi ** 2 - lo ** 2) / 2
                            + c2 * (hi ** 3 - lo ** 3) / 3)
        return float(total)

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
