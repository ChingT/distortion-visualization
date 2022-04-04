import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def main():
    camera_params = set_sliders()
    render(camera_params)
    print_text()


def set_sliders():
    st.sidebar.button("Reset", on_click=reset_sliders)

    st.sidebar.subheader("Radial distortion coefficients")
    args = dict(min_value=-1.0, max_value=1.0, step=0.01, value=0.0)
    k1 = st.sidebar.slider("k1", key="k1", **args)
    k2 = st.sidebar.slider("k2", key="k2", **args)
    k3 = st.sidebar.slider("k3", key="k3", **args)
    k4 = st.sidebar.slider("k4", key="k4", **args)
    k5 = st.sidebar.slider("k5", key="k5", **args)
    k6 = st.sidebar.slider("k6", key="k6", **args)

    st.sidebar.subheader("Tangential distortion coefficients")
    args = dict(min_value=-0.1, max_value=0.1, step=0.01, value=0.0)
    p1 = st.sidebar.slider("p1", key="p1", **args)
    p2 = st.sidebar.slider("p2", key="p2", **args)

    fx, fy, cx, cy = 3000, 3000, 2000, 1500
    camera_params = [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]
    return camera_params


def reset_sliders(value=0):
    for key in ["k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"]:
        st.session_state[key] = value


def render(camera_params):
    st.subheader("Distorted points on image plane")

    camera_matrix, dist_coefs = convert_camera_params(camera_params)
    points_3d = get_points_3d()
    points_2d_proj = project_points(points_3d, camera_matrix, dist_coefs)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(points_2d_proj[:, 0], points_2d_proj[:, 1], ls="", marker=".")

    width = camera_matrix[0][2] * 2
    height = camera_matrix[1][2] * 2
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def convert_camera_params(camera_params):
    camera_params = np.asarray(camera_params, dtype=np.float64)
    dist_coefs = np.zeros(8, dtype=np.float64)

    # Full OpenCV camera model: [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]
    focal_length = camera_params[:2]
    principal_point = camera_params[2:4]
    dist_coefs[: len(camera_params[4:])] = camera_params[4:]

    camera_matrix = np.eye(3, dtype=np.float64)
    camera_matrix[[0, 1], [0, 1]] = focal_length
    camera_matrix[[0, 1], [2, 2]] = principal_point
    return camera_matrix, dist_coefs


def get_points_3d(lim=0.75, n=20):
    x = np.linspace(-lim, lim, n)
    y = np.linspace(-lim, lim, n)
    xx, yy = np.meshgrid(x, y)
    zz = np.ones(xx.size)
    return np.vstack((xx.ravel(), yy.ravel(), zz)).T


def project_points(points_3d, camera_matrix, dist_coefs):
    points_2d, _ = cv2.projectPoints(
        points_3d.reshape(1, -1, 3),
        np.zeros((1, 1, 3)),
        np.zeros((1, 1, 3)),
        camera_matrix,
        dist_coefs,
    )
    return points_2d.reshape(-1, 2)



def print_text():
    st.latex(
        r"""
\begin{array}{l} x'' = x' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + 2 p_1 x' y' + p_2(r^2 + 2 x'^2)  \\
y'' = y' \frac{1 + k_1 r^2 + k_2 r^4 + k_3 r^6}{1 + k_4 r^2 + k_5 r^4 + k_6 r^6} + p_1 (r^2 + 2 y'^2) + 2 p_2 x' y' \\
\end{array}
"""
    )
    st.markdown(
        "Checkout [OpenCV documentation]"
        "(https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html) "
        "for the complete distortion formula."
    )

if __name__ == "__main__":
    main()
