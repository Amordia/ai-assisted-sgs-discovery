from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

FloatArray = NDArray[np.float64]

_DEFAULT_UNIFORM_DX = 2.0 * np.pi / 1024.0
_CHANNEL_X_SPACING = 8.0 * np.pi / 2048.0
_CHANNEL_Z_SPACING = 3.0 * np.pi / 1536.0
_DEFAULT_DT = 0.002


@dataclass
class FilteredVolume:
    h5_path: str
    boundary_mode: str
    filter_width: float
    u_t1: FloatArray | None
    u_t2: FloatArray
    u_t3: FloatArray | None
    u_bar_t1: FloatArray | None
    u_bar_t2: FloatArray
    u_bar_t3: FloatArray | None
    x_coords: FloatArray
    y_coords: FloatArray | None
    z_coords: FloatArray
    axis_coords: tuple[float | FloatArray, float | FloatArray, float | FloatArray]
    axis_widths: tuple[FloatArray, FloatArray, FloatArray]
    delta_eff: FloatArray
    grid_shape: tuple[int, int, int]


@dataclass
class FilteredSequence:
    h5_path: str
    boundary_mode: str
    filter_width: float
    raw_frames: list[FloatArray]
    filtered_frames: list[FloatArray]
    x_coords: FloatArray
    y_coords: FloatArray | None
    z_coords: FloatArray
    axis_coords: tuple[float | FloatArray, float | FloatArray, float | FloatArray]
    axis_widths: tuple[FloatArray, FloatArray, FloatArray]
    delta_eff: FloatArray
    grid_shape: tuple[int, int, int]
    dt: float
    timepoints: FloatArray


def _filter_vector_field(u: np.ndarray, sigma: float, mode: str) -> FloatArray:
    out = np.zeros_like(u, dtype=np.float64)
    for i in range(3):
        out[..., i] = gaussian_filter(u[..., i], sigma=sigma, mode=mode)
    return out


def _infer_coords(
    Z: int,
    Y: int,
    X: int,
    x_coords: np.ndarray | None,
    y_coords: np.ndarray | None,
    z_coords: np.ndarray | None,
) -> tuple[FloatArray, FloatArray | None, FloatArray]:
    if x_coords is not None and z_coords is not None:
        x = np.asarray(x_coords, dtype=np.float64)
        z = np.asarray(z_coords, dtype=np.float64)
        y = None if y_coords is None else np.asarray(y_coords, dtype=np.float64)
        return x, y, z

    if y_coords is None:
        coords = np.arange(X, dtype=np.float64) * _DEFAULT_UNIFORM_DX
        return coords.copy(), None, coords.copy()

    x_coords = np.arange(X, dtype=np.float64) * _CHANNEL_X_SPACING
    z_coords = np.arange(Z, dtype=np.float64) * _CHANNEL_Z_SPACING
    return x_coords, np.asarray(y_coords, dtype=np.float64), z_coords


def _axis_widths(
    x_coords: FloatArray,
    y_coords: FloatArray | None,
    z_coords: FloatArray,
    grid_shape: tuple[int, int, int],
    filter_width: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    Z, Y, X = grid_shape
    dx_1d = np.abs(np.gradient(x_coords)).astype(np.float64)
    dz_1d = np.abs(np.gradient(z_coords)).astype(np.float64)
    if y_coords is None:
        dy_1d = np.full(Y, _DEFAULT_UNIFORM_DX, dtype=np.float64)
    else:
        dy_1d = np.abs(np.gradient(y_coords)).astype(np.float64)

    dx = np.broadcast_to(dx_1d[None, None, :], (Z, Y, X)).copy()
    dy = np.broadcast_to(dy_1d[None, :, None], (Z, Y, X)).copy()
    dz = np.broadcast_to(dz_1d[:, None, None], (Z, Y, X)).copy()
    delta_eff = filter_width * np.cbrt(dx * dy * dz)
    return dx, dy, dz, delta_eff


def load_filtered_volume(
    h5_path: str,
    filter_width: float,
    boundary_mode: str,
) -> FilteredVolume:
    with h5py.File(h5_path, "r") as f:
        u_t2 = np.asarray(f["Velocity_t2"], dtype=np.float64)
        u_t1 = np.asarray(f["Velocity_t1"], dtype=np.float64) if "Velocity_t1" in f else None
        u_t3 = np.asarray(f["Velocity_t3"], dtype=np.float64) if "Velocity_t3" in f else None
        x_coords = np.asarray(f["x_coords"], dtype=np.float64) if "x_coords" in f else None
        y_coords = np.asarray(f["y_coords"], dtype=np.float64) if "y_coords" in f else None
        z_coords = np.asarray(f["z_coords"], dtype=np.float64) if "z_coords" in f else None

    Z, Y, X, _ = u_t2.shape
    x_coords, y_coords, z_coords = _infer_coords(
        Z=Z,
        Y=Y,
        X=X,
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords,
    )

    if y_coords is None and x_coords is not None and z_coords is not None:
        axis_coords = (z_coords, float(np.abs(np.gradient(x_coords)).mean()), x_coords)
    elif y_coords is None:
        axis_coords = (
            float(_DEFAULT_UNIFORM_DX),
            float(_DEFAULT_UNIFORM_DX),
            float(_DEFAULT_UNIFORM_DX),
        )
    else:
        axis_coords = (z_coords, y_coords, x_coords)

    dx, dy, dz, delta_eff = _axis_widths(
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords,
        grid_shape=(Z, Y, X),
        filter_width=filter_width,
    )

    return FilteredVolume(
        h5_path=h5_path,
        boundary_mode=boundary_mode,
        filter_width=filter_width,
        u_t1=u_t1,
        u_t2=u_t2,
        u_t3=u_t3,
        u_bar_t1=_filter_vector_field(u_t1, filter_width, boundary_mode) if u_t1 is not None else None,
        u_bar_t2=_filter_vector_field(u_t2, filter_width, boundary_mode),
        u_bar_t3=_filter_vector_field(u_t3, filter_width, boundary_mode) if u_t3 is not None else None,
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords,
        axis_coords=axis_coords,
        axis_widths=(dx, dy, dz),
        delta_eff=delta_eff,
        grid_shape=(Z, Y, X),
    )


def load_filtered_sequence(
    h5_path: str,
    filter_width: float,
    boundary_mode: str,
) -> FilteredSequence:
    with h5py.File(h5_path, "r") as f:
        x_coords = np.asarray(f["x_coords"], dtype=np.float64) if "x_coords" in f else None
        y_coords = np.asarray(f["y_coords"], dtype=np.float64) if "y_coords" in f else None
        z_coords = np.asarray(f["z_coords"], dtype=np.float64) if "z_coords" in f else None
        frame_keys = sorted(k for k in f.keys() if k.startswith("Velocity_t"))
        raw_frames = [np.asarray(f[key], dtype=np.float64) for key in frame_keys]
        dt = float(f.attrs.get("dt", _DEFAULT_DT))
        timepoints = np.asarray(f.attrs.get("timepoints", np.arange(len(frame_keys), dtype=np.float64) * dt), dtype=np.float64)

    if not raw_frames:
        raise ValueError(f"no Velocity_t* datasets found in {h5_path}")

    Z, Y, X, _ = raw_frames[0].shape
    x_coords, y_coords, z_coords = _infer_coords(
        Z=Z,
        Y=Y,
        X=X,
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords,
    )
    if y_coords is None and x_coords is not None and z_coords is not None:
        axis_coords = (z_coords, float(np.abs(np.gradient(x_coords)).mean()), x_coords)
    elif y_coords is None:
        axis_coords = (float(_DEFAULT_UNIFORM_DX), float(_DEFAULT_UNIFORM_DX), float(_DEFAULT_UNIFORM_DX))
    else:
        axis_coords = (z_coords, y_coords, x_coords)

    dx, dy, dz, delta_eff = _axis_widths(
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords,
        grid_shape=(Z, Y, X),
        filter_width=filter_width,
    )
    filtered_frames = [_filter_vector_field(frame, filter_width, boundary_mode) for frame in raw_frames]

    return FilteredSequence(
        h5_path=h5_path,
        boundary_mode=boundary_mode,
        filter_width=filter_width,
        raw_frames=raw_frames,
        filtered_frames=filtered_frames,
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords,
        axis_coords=axis_coords,
        axis_widths=(dx, dy, dz),
        delta_eff=delta_eff,
        grid_shape=(Z, Y, X),
        dt=dt,
        timepoints=timepoints,
    )


def gradient_axis(field: np.ndarray, coords: float | FloatArray, axis: int) -> FloatArray:
    edge_order = 2 if field.shape[axis] >= 3 else 1
    return np.gradient(field, coords, axis=axis, edge_order=edge_order)


def gradient3(
    field: np.ndarray,
    axis_coords: tuple[float | FloatArray, float | FloatArray, float | FloatArray],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    return (
        gradient_axis(field, axis_coords[0], axis=0),
        gradient_axis(field, axis_coords[1], axis=1),
        gradient_axis(field, axis_coords[2], axis=2),
    )


def velocity_gradient(
    u: np.ndarray,
    axis_coords: tuple[float | FloatArray, float | FloatArray, float | FloatArray],
) -> FloatArray:
    grad_u = np.zeros(u.shape[:3] + (3, 3), dtype=np.float64)
    for i in range(3):
        grad_z, grad_y, grad_x = gradient3(u[..., i], axis_coords)
        grad_u[..., i, 0] = grad_x
        grad_u[..., i, 1] = grad_y
        grad_u[..., i, 2] = grad_z
    return grad_u


def strain_rotation(grad_u: np.ndarray) -> tuple[FloatArray, FloatArray]:
    S = 0.5 * (grad_u + np.swapaxes(grad_u, -1, -2))
    Omega = 0.5 * (grad_u - np.swapaxes(grad_u, -1, -2))
    return S, Omega


def wale_tensor(grad_u: np.ndarray) -> FloatArray:
    g_sq = np.matmul(grad_u, grad_u)
    S_d = 0.5 * (g_sq + np.swapaxes(g_sq, -1, -2))
    trace_g_sq = np.trace(g_sq, axis1=-2, axis2=-1)
    for i in range(3):
        S_d[..., i, i] -= trace_g_sq / 3.0
    return S_d


def topological_vectors(u: np.ndarray, S: np.ndarray, Omega: np.ndarray) -> tuple[FloatArray, FloatArray, FloatArray]:
    omega = np.zeros(u.shape[:3] + (3,), dtype=np.float64)
    omega[..., 0] = 2.0 * Omega[..., 2, 1]
    omega[..., 1] = 2.0 * Omega[..., 0, 2]
    omega[..., 2] = 2.0 * Omega[..., 1, 0]
    W = np.matmul(S, omega[..., None])[..., 0]
    h = np.sum(u * omega, axis=-1, keepdims=True)
    return omega, W, h


def laplacian_tensor(
    tensor: np.ndarray,
    axis_coords: tuple[float | FloatArray, float | FloatArray, float | FloatArray],
) -> FloatArray:
    lap = np.zeros_like(tensor, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            grad_z, grad_y, grad_x = gradient3(tensor[..., i, j], axis_coords)
            grad_xx = gradient_axis(grad_x, axis_coords[2], axis=2)
            grad_yy = gradient_axis(grad_y, axis_coords[1], axis=1)
            grad_zz = gradient_axis(grad_z, axis_coords[0], axis=0)
            lap[..., i, j] = grad_xx + grad_yy + grad_zz
    return lap


def jaumann_rate(
    u: np.ndarray,
    S: np.ndarray,
    Omega: np.ndarray,
    axis_coords: tuple[float | FloatArray, float | FloatArray, float | FloatArray],
    prev_u: np.ndarray | None = None,
    dt: float = _DEFAULT_DT,
) -> FloatArray:
    dS_dt = np.zeros_like(S, dtype=np.float64)
    if prev_u is not None:
        prev_grad = velocity_gradient(prev_u, axis_coords)
        prev_S, _ = strain_rotation(prev_grad)
        dS_dt = (S - prev_S) / dt

    dS_dx = np.zeros(S.shape[:3] + (3, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            grad_z, grad_y, grad_x = gradient3(S[..., i, j], axis_coords)
            dS_dx[..., i, j, 0] = grad_x
            dS_dx[..., i, j, 1] = grad_y
            dS_dx[..., i, j, 2] = grad_z

    convective = np.einsum("...k,...ijk->...ij", u, dS_dx)
    spin = np.matmul(S, Omega) - np.matmul(Omega, S)
    return dS_dt + convective + spin


def leonard_stress(u: np.ndarray, sigma_extra: float, mode: str) -> FloatArray:
    u_hat = _filter_vector_field(u, sigma_extra, mode)
    L = np.zeros(u.shape[:3] + (3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            L[..., i, j] = gaussian_filter(u[..., i] * u[..., j], sigma=sigma_extra, mode=mode) - u_hat[..., i] * u_hat[..., j]
    return L


def feature_bundle(
    u: np.ndarray,
    axis_coords: tuple[float | FloatArray, float | FloatArray, float | FloatArray],
    mode: str,
    prev_u: np.ndarray | None = None,
    dt: float = _DEFAULT_DT,
    sigma_extra: float | None = None,
) -> dict[str, FloatArray]:
    grad_u = velocity_gradient(u, axis_coords)
    S, Omega = strain_rotation(grad_u)
    S_d = wale_tensor(grad_u)
    omega_vec, W_vec, h_scalar = topological_vectors(u, S, Omega)
    Lap_S = laplacian_tensor(S, axis_coords)
    S_j = jaumann_rate(u, S, Omega, axis_coords, prev_u=prev_u, dt=dt)
    bundle: dict[str, FloatArray] = {
        "grad_u": grad_u,
        "S": S,
        "Omega": Omega,
        "S_d": S_d,
        "omega_vec": omega_vec,
        "W_vec": W_vec,
        "h_scalar": h_scalar,
        "Lap_S": Lap_S,
        "S_j": S_j,
    }
    if sigma_extra is not None:
        bundle["L"] = leonard_stress(u, sigma_extra=sigma_extra, mode=mode)
    return bundle


def tensor_gaussian_filter(tensor: np.ndarray, sigma: float, mode: str) -> FloatArray:
    out = np.zeros_like(tensor, dtype=np.float64)
    for i in range(tensor.shape[-2]):
        for j in range(tensor.shape[-1]):
            out[..., i, j] = gaussian_filter(tensor[..., i, j], sigma=sigma, mode=mode)
    return out


def deviatoric(tau: np.ndarray) -> FloatArray:
    tau = np.asarray(tau, dtype=np.float64)
    tr = np.trace(tau, axis1=-2, axis2=-1)[..., None, None] / 3.0
    return tau - tr * np.eye(3, dtype=np.float64)


def tau_bardina(L: np.ndarray) -> FloatArray:
    return np.asarray(L, dtype=np.float64)


def tau_smagorinsky(S: np.ndarray, delta_eff: np.ndarray, c_s: float = 0.17) -> FloatArray:
    s_sq = np.sum(S * S, axis=(-2, -1))
    mag_S = np.sqrt(2.0 * np.maximum(s_sq, 0.0))
    nu_t = (c_s * delta_eff) ** 2 * mag_S
    return -2.0 * nu_t[..., None, None] * S


def tau_wale(S: np.ndarray, S_d: np.ndarray, delta_eff: np.ndarray, c_w: float = 0.325) -> FloatArray:
    s_sq = np.sum(S * S, axis=(-2, -1))
    sd_sq = np.sum(S_d * S_d, axis=(-2, -1))
    eps = 1.0e-30
    numerator = np.power(np.maximum(sd_sq, 0.0) + eps, 1.5)
    denominator = np.power(np.maximum(s_sq, 0.0) + eps, 2.5) + np.power(np.maximum(sd_sq, 0.0) + eps, 1.25)
    nu_t = (c_w * delta_eff) ** 2 * numerator / denominator
    return -2.0 * nu_t[..., None, None] * S


def tau_vreman(
    grad_u: np.ndarray,
    S: np.ndarray,
    axis_widths: tuple[np.ndarray, np.ndarray, np.ndarray],
    c_v: float = 2.5 * 0.17 ** 2,
) -> FloatArray:
    a = np.swapaxes(grad_u, -1, -2)
    beta = np.zeros_like(S, dtype=np.float64)
    for axis_idx, delta_axis in enumerate(axis_widths):
        col = a[..., axis_idx, :]
        beta += (delta_axis[..., None, None] ** 2) * np.einsum("...i,...j->...ij", col, col)

    b11 = beta[..., 0, 0]
    b22 = beta[..., 1, 1]
    b33 = beta[..., 2, 2]
    b12 = beta[..., 0, 1]
    b13 = beta[..., 0, 2]
    b23 = beta[..., 1, 2]
    bbeta = (b11 * b22 - b12 * b12) + (b11 * b33 - b13 * b13) + (b22 * b33 - b23 * b23)
    a_sq = np.sum(a * a, axis=(-2, -1))
    nu_t = c_v * np.sqrt(np.maximum(bbeta, 0.0) / (a_sq + 1.0e-30))
    return -2.0 * nu_t[..., None, None] * S


def tau_amd(
    grad_u: np.ndarray,
    S: np.ndarray,
    axis_widths: tuple[np.ndarray, np.ndarray, np.ndarray],
    c_a: float = 0.3,
) -> FloatArray:
    weighted_outer = np.zeros_like(S, dtype=np.float64)
    for axis_idx, delta_axis in enumerate(axis_widths):
        gk = delta_axis[..., None] * grad_u[..., :, axis_idx]
        weighted_outer += np.einsum("...i,...j->...ij", gk, gk)

    numerator = -np.einsum("...ij,...ij->...", weighted_outer, S)
    denominator = np.sum(grad_u * grad_u, axis=(-2, -1)) + 1.0e-30
    nu_t = c_a * np.maximum(numerator, 0.0) / denominator
    return -2.0 * nu_t[..., None, None] * S


def tau_dynamic_smagorinsky(
    u: np.ndarray,
    S: np.ndarray,
    delta_eff: np.ndarray,
    axis_coords: tuple[float | FloatArray, float | FloatArray, float | FloatArray],
    mode: str,
    test_sigma: float = np.sqrt(3.0),
    plane_average: bool = False,
) -> tuple[FloatArray, FloatArray]:
    u_hat = _filter_vector_field(u, test_sigma, mode)
    grad_hat = velocity_gradient(u_hat, axis_coords)
    S_hat, _ = strain_rotation(grad_hat)

    mag_S = np.sqrt(2.0 * np.maximum(np.sum(S * S, axis=(-2, -1)), 0.0))
    mag_S_hat = np.sqrt(2.0 * np.maximum(np.sum(S_hat * S_hat, axis=(-2, -1)), 0.0))
    delta_hat = 2.0 * delta_eff

    modeled = delta_eff[..., None, None] ** 2 * mag_S[..., None, None] * S
    modeled_hat = tensor_gaussian_filter(modeled, sigma=test_sigma, mode=mode)
    test_level = delta_hat[..., None, None] ** 2 * mag_S_hat[..., None, None] * S_hat

    M = 2.0 * (modeled_hat - test_level)
    L = deviatoric(leonard_stress(u, sigma_extra=test_sigma, mode=mode))
    M = deviatoric(M)

    if plane_average:
        numerator = np.mean(L * M, axis=(0, 2, 3, 4))
        denominator = np.mean(M * M, axis=(0, 2, 3, 4)) + 1.0e-30
        coeff = np.maximum(numerator / denominator, 0.0)
        coeff_field = coeff[None, :, None]
    else:
        numerator = float(np.mean(L * M))
        denominator = float(np.mean(M * M) + 1.0e-30)
        coeff_field = np.array(max(numerator / denominator, 0.0), dtype=np.float64)

    tau = -2.0 * coeff_field[..., None, None] * delta_eff[..., None, None] ** 2 * mag_S[..., None, None] * S
    return tau, np.asarray(coeff_field, dtype=np.float64)


def flatten_tensor_field(field: np.ndarray) -> FloatArray:
    return np.asarray(field, dtype=np.float64).reshape(-1, 3, 3)


def flatten_vector_field(field: np.ndarray) -> FloatArray:
    return np.asarray(field, dtype=np.float64).reshape(-1, 3)


def flatten_scalar_field(field: np.ndarray) -> FloatArray:
    return np.asarray(field, dtype=np.float64).reshape(-1)


def divergence_tensor(
    tau: np.ndarray,
    axis_coords: tuple[float | FloatArray, float | FloatArray, float | FloatArray],
) -> FloatArray:
    out = np.zeros(tau.shape[:3] + (3,), dtype=np.float64)
    for i in range(3):
        grad_x = gradient_axis(tau[..., i, 0], axis_coords[2], axis=2)
        grad_y = gradient_axis(tau[..., i, 1], axis_coords[1], axis=1)
        grad_z = gradient_axis(tau[..., i, 2], axis_coords[0], axis=0)
        out[..., i] = grad_x + grad_y + grad_z
    return out


def volume_mean_kinetic_energy(u: np.ndarray) -> float:
    return float(0.5 * np.mean(np.sum(np.asarray(u, dtype=np.float64) ** 2, axis=-1)))


def isotropic_shell_spectrum(u: np.ndarray, box_lengths: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    Z, Y, X, _ = u.shape
    kz = 2.0 * np.pi * np.fft.fftfreq(Z, d=box_lengths[0] / Z)
    ky = 2.0 * np.pi * np.fft.fftfreq(Y, d=box_lengths[1] / Y)
    kx = 2.0 * np.pi * np.fft.fftfreq(X, d=box_lengths[2] / X)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    uhat = np.fft.fftn(u, axes=(0, 1, 2))
    energy_density = 0.5 * np.sum(np.abs(uhat) ** 2, axis=-1) / (X * Y * Z) ** 2
    shell_index = np.rint(kmag * min(box_lengths) / (2.0 * np.pi)).astype(int)
    max_shell = int(shell_index.max())
    spectrum = np.zeros(max_shell + 1, dtype=np.float64)
    counts = np.zeros(max_shell + 1, dtype=np.int64)
    for shell in range(max_shell + 1):
        mask = shell_index == shell
        spectrum[shell] = float(np.sum(energy_density[mask]))
        counts[shell] = int(np.count_nonzero(mask))
    k_shell = np.arange(max_shell + 1, dtype=np.float64)
    valid = counts > 0
    return k_shell[valid], spectrum[valid]


def spectrum_corr(reference: np.ndarray, prediction: np.ndarray) -> float:
    n = min(reference.size, prediction.size)
    if n < 3:
        return float("nan")
    ref = reference[:n]
    pred = prediction[:n]
    if np.std(ref) < 1.0e-15 or np.std(pred) < 1.0e-15:
        return float("nan")
    return float(np.corrcoef(ref, pred)[0, 1])


def box_lengths_from_volume(volume: FilteredVolume) -> tuple[float, float, float]:
    z_len = float(volume.z_coords[-1] - volume.z_coords[0] + np.abs(np.gradient(volume.z_coords)).mean())
    if volume.y_coords is None:
        y_len = float(volume.x_coords[-1] - volume.x_coords[0] + np.abs(np.gradient(volume.x_coords)).mean())
    else:
        y_len = float(volume.y_coords[-1] - volume.y_coords[0] + np.abs(np.gradient(volume.y_coords)).mean())
    x_len = float(volume.x_coords[-1] - volume.x_coords[0] + np.abs(np.gradient(volume.x_coords)).mean())
    return z_len, y_len, x_len
