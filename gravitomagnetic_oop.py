import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator, interp1d


class FieldPlotter:
    """
    Plot 2D slices of 3D cosmological fields.
    """

    @staticmethod
    def plot_field(field, field_name, slice_idx=None, linthresh=1.0, cmap="hot", vmin=None, vmax=None):
        """
        Plot a 2D slice of a 3D cosmological field (density, overdensity, or momentum).

        Parameters
        ----------
        field : ndarray
            2D or 3D array to plot.
        field_name : str
            Name of the field: 'delta', 'density', 'q_x', 'q_y', 'q_z'
        slice_idx : int, optional
            Slice index along the 3rd axis if field is 3D. Default: middle slice.
        linthresh : float
            Linear threshold for SymLogNorm (delta and momentum fields).
        cmap : str
            Colormap to use.
        vmin, vmax : float, optional
            Color limits. If None, they will be determined automatically.
        """
        if field.ndim == 3:
            if slice_idx is None:
                slice_idx = field.shape[2] // 2
            data = field[:, :, slice_idx].copy()
        else:
            data = field.copy()

        if field_name == "density":
            data = data + 1
            label = r"$\rho/\bar{\rho}$"
        elif field_name == "delta":
            label = r"$\delta$"
        elif field_name in ["q_x", "q_y", "q_z"]:
            label = rf"${field_name}$"
        else:
            raise ValueError("field_name must be 'delta', 'density', 'q_x', 'q_y', or 'q_z'")

        norm = SymLogNorm(
            linthresh=linthresh,
            vmin=vmin if vmin is not None else np.min(data),
            vmax=vmax if vmax is not None else np.max(data),
        )

        fig, ax = plt.subplots()
        cax = ax.imshow(data, cmap=cmap, norm=norm)
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(label, fontsize=14)
        ax.set_title(f"{field_name} field slice {slice_idx}")
        plt.show()


class PowerSpectrumInterpolators:
    """
    Interpolators for the matter and momentum power spectra in the notebook.
    """

    def __init__(
        self,
        *,
        kk=None,
        pk_m_class_const_z=None,
        z_list=None,
        pk_m_class_varz_4interp=None,
        k_m=None,
        pk_m=None,
        k_omega=None,
        pk_b=None,
    ):
        self._pk_m_class_constz_interp = None
        self._pk_m_class_varz_interp = None
        self._pk_matter_int = None
        self._pk_b_int = None

        if kk is not None and pk_m_class_const_z is not None:
            self._pk_m_class_constz_interp = interp1d(
                np.log(kk), np.log(pk_m_class_const_z), kind="cubic"
            )

        if kk is not None and z_list is not None and pk_m_class_varz_4interp is not None:
            self._pk_m_class_varz_interp = RegularGridInterpolator(
                (np.log(kk), z_list),
                np.log(pk_m_class_varz_4interp),
                method="cubic",
                bounds_error=False,
                fill_value=None,
            )

        if k_m is not None and pk_m is not None:
            self._pk_matter_int = interp1d(np.log(k_m), np.log(pk_m), kind="cubic")

        if k_omega is not None and pk_b is not None:
            self._pk_b_int = interp1d(np.log(k_omega), np.log(pk_b), kind="cubic")

    def pk_m_class_constz(self, k):
        if self._pk_m_class_constz_interp is None:
            raise ValueError("pk_m_class_const_z interpolator is not initialized")
        return np.exp(self._pk_m_class_constz_interp(np.log(k)))

    def pk_m_class_varz(self, k, z=None):
        if self._pk_m_class_varz_interp is None:
            raise ValueError("pk_m_class_varz interpolator is not initialized")
        if z is None:
            k, z = k
        return np.exp(self._pk_m_class_varz_interp((np.log(k), z)))

    def pk_matter(self, k):
        if self._pk_matter_int is None:
            raise ValueError("pk_matter interpolator is not initialized")
        return np.exp(self._pk_matter_int(np.log(k)))

    def pk_q(self, k):
        if self._pk_b_int is None:
            raise ValueError("pk_b interpolator is not initialized")
        return np.exp(self._pk_b_int(np.log(k))) * k**4


class AngularPowerSpectrumCalculator:
    """
    Compute C_ell for arbitrary kernels using notebook-style inputs.
    """

    def __init__(self, *, chi_of_z, z_of_chi, hubble, c_light):
        self._chi_of_z = chi_of_z
        self._z_of_chi = z_of_chi
        self._hubble = hubble
        self._c_light = c_light

    def c_ell_xy(
        self,
        chi_s,
        ell,
        kmin,
        kmax,
        pk,
        kernel_x,
        kernel_y,
        *,
        z_min=1e-5,
        pk_evol=False,
    ):
        """
        Compute C_ell^{XY} for two arbitrary kernels X and Y.

        Parameters
        ----------
        chi_s : float
            Source comoving distance.
        ell : float
            Multipole moment.
        kmin, kmax : float
            Limits of k to restrict the chi integration range.
        pk : function
            Power spectrum P(k) or P(k,z) depending on pk_evol.
        kernel_x, kernel_y : function
            Kernel functions of the form kernel(chi, chi_s) or kernel(chi).
        z_min : float
            Minimum chi integration limit.
        pk_evol : bool
            If True, pk takes two arguments: pk(k,z).
        """
        z_s = self._z_of_chi(chi_s)
        z_grid = np.linspace(z_min, z_s, 1000)
        chi_grid = self._chi_of_z(z_grid)

        mask = (ell / chi_grid < kmax) & (ell / chi_grid > kmin)
        z_masked = z_grid[mask]
        r = chi_grid[mask]

        if pk_evol:
            c_ell_int = pk(ell / r, z_masked) / r**2
        else:
            c_ell_int = pk(ell / r) / r**2

        try:
            kx = kernel_x(r, chi_s)
        except TypeError:
            kx = kernel_x(r)
        try:
            ky = kernel_y(r, chi_s)
        except TypeError:
            ky = kernel_y(r)

        c_ell_int *= kx * ky
        c_ell_int *= self._c_light / self._hubble(z_masked)

        return simpson(c_ell_int, x=z_masked)
