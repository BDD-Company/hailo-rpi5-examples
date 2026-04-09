"""Interactive 3D step-by-step flight visualizer for drone telemetry."""

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from debug_telemetry_position import FramePose, parse_telemetry_log, LOG_PATH
from telemetry_position import Quaternion, rotate_frd_to_ned

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

# Simple quadrotor body shape in body frame (FRD):
#   4 arms of length ARM_LEN along +X, -X, +Y, -Y (forward, back, right, left)
ARM_LEN = 0.8  # meters (visual scale)
BODY_VERTS_FRD = np.array([
    # arm tips: forward, back, right, left
    [ARM_LEN,  0.0,       0.0],
    [-ARM_LEN, 0.0,       0.0],
    [0.0,      ARM_LEN,   0.0],
    [0.0,     -ARM_LEN,   0.0],
    # center
    [0.0,      0.0,       0.0],
    # nose marker (small triangle on forward arm)
    [ARM_LEN * 1.15,  0.08, 0.0],
    [ARM_LEN * 1.15, -0.08, 0.0],
])


def quat_rotate_array(q: Quaternion, pts: np.ndarray) -> np.ndarray:
    """Rotate an (N,3) array of FRD vectors to NED using quaternion."""
    out = np.empty_like(pts)
    for i in range(pts.shape[0]):
        n, e, d = rotate_frd_to_ned(q, pts[i, 0], pts[i, 1], pts[i, 2])
        out[i] = [n, e, d]
    return out


def ned_to_plot(n: float, e: float, d: float) -> tuple[float, float, float]:
    """Convert NED to plot coords (X=East, Y=North, Z=Up)."""
    return e, n, -d


def ned_array_to_plot(arr: np.ndarray) -> np.ndarray:
    """Convert (N,3) NED array to plot coords."""
    return arr[:, [1, 0, 2]] * np.array([1, 1, -1])


# ---------------------------------------------------------------------------
# Build per-frame data arrays for fast access
# ---------------------------------------------------------------------------

def precompute(frames: list[FramePose]):
    n = len(frames)
    pos = np.zeros((n, 3))        # plot coords
    vel_vec = np.zeros((n, 3))    # velocity in plot coords (NED already)
    acc_vec = np.zeros((n, 3))    # acceleration rotated to NED, then plot
    mag_vec = np.zeros((n, 3))    # magnetic field rotated to NED, then plot

    for i, fp in enumerate(frames):
        p = fp.pose
        pos[i] = ned_to_plot(p.position.north_m, p.position.east_m, p.position.down_m)

        # Velocity is already in NED frame (child_frame_id=1=LOCAL_NED)
        vel_vec[i] = ned_to_plot(
            p.velocity.north_m_s, p.velocity.east_m_s, p.velocity.down_m_s,
        )

        if p.acceleration:
            an, ae, ad = rotate_frd_to_ned(
                p.quaternion,
                p.acceleration.forward_m_s2,
                p.acceleration.right_m_s2,
                p.acceleration.down_m_s2,
            )
            acc_vec[i] = ned_to_plot(an, ae, ad)

        if p.magnetic_field:
            mn, me, md = rotate_frd_to_ned(
                p.quaternion,
                p.magnetic_field.forward_gauss,
                p.magnetic_field.right_gauss,
                p.magnetic_field.down_gauss,
            )
            mag_vec[i] = ned_to_plot(mn, me, md)

    return pos, vel_vec, acc_vec, mag_vec


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

VECTOR_SCALE = {
    "vel": 3.0,     # m/s  -> visual metres
    "acc": 0.3,     # m/s² -> visual metres (gravity ~10, so 0.3*10=3m arrow)
    "mag": 8.0,     # gauss -> visual metres (values ~0.3, so 8*0.3=2.4m arrow)
}


def run(frames: list[FramePose]) -> None:
    pos, vel_vec, acc_vec, mag_vec = precompute(frames)

    # Relative position: subtract first frame so trail starts near origin
    origin = pos[0].copy()
    pos_rel = pos - origin

    fig = plt.figure(figsize=(14, 9))
    fig.subplots_adjust(bottom=0.18)
    ax = fig.add_subplot(111, projection="3d")

    # ---------- static trail ----------
    trail_line, = ax.plot(pos_rel[:, 0], pos_rel[:, 1], pos_rel[:, 2],
                          "-", color="0.70", linewidth=0.8, zorder=1)
    trail_past, = ax.plot([], [], [], "-", color="steelblue", linewidth=1.5, zorder=2)

    # ---------- current position marker ----------
    drone_marker, = ax.plot([], [], [], "ko", markersize=5, zorder=5)

    # ---------- drone body (quad cross + nose) ----------
    arm1_line, = ax.plot([], [], [], "-", color="black", linewidth=2.5, zorder=6)
    arm2_line, = ax.plot([], [], [], "-", color="black", linewidth=2.5, zorder=6)
    nose_poly_coll = None  # will be replaced each frame

    # ---------- quiver arrows (velocity, acceleration, magnetic) ----------
    quiv_vel = None
    quiv_acc = None
    quiv_mag = None

    # ---------- text ----------
    info_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=9,
                          verticalalignment="top", fontfamily="monospace")

    # ---------- axis setup ----------
    pad = 3.0
    all_min = pos_rel.min(axis=0) - pad
    all_max = pos_rel.max(axis=0) + pad
    # Make axes equal-ish
    ranges = all_max - all_min
    max_range = ranges.max()
    mid = (all_min + all_max) / 2
    ax.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
    ax.set_ylim(mid[1] - max_range / 2, mid[1] + max_range / 2)
    ax.set_zlim(mid[2] - max_range / 2, mid[2] + max_range / 2)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title("Drone Flight Telemetry")
    ax.view_init(elev=30, azim=45)

    # ---------- update function ----------
    def update(idx: int):
        nonlocal quiv_vel, quiv_acc, quiv_mag, nose_poly_coll
        idx = int(idx)
        fp = frames[idx]
        p = fp.pose
        cx, cy, cz = pos_rel[idx]

        # Past trail
        trail_past.set_data_3d(pos_rel[:idx + 1, 0], pos_rel[:idx + 1, 1], pos_rel[:idx + 1, 2])

        # Drone marker
        drone_marker.set_data_3d([cx], [cy], [cz])

        # Drone body cross — rotate body verts by quaternion, translate to position
        rotated = quat_rotate_array(p.quaternion, BODY_VERTS_FRD)
        plot_verts = ned_array_to_plot(rotated)
        plot_verts += np.array([cx, cy, cz])
        # arm1: forward-back (verts 0,1), arm2: right-left (verts 2,3)
        arm1_line.set_data_3d([plot_verts[0, 0], plot_verts[1, 0]],
                              [plot_verts[0, 1], plot_verts[1, 1]],
                              [plot_verts[0, 2], plot_verts[1, 2]])
        arm2_line.set_data_3d([plot_verts[2, 0], plot_verts[3, 0]],
                              [plot_verts[2, 1], plot_verts[3, 1]],
                              [plot_verts[2, 2], plot_verts[3, 2]])
        # Nose triangle
        if nose_poly_coll is not None:
            nose_poly_coll.remove()
        nose_tri = [plot_verts[[0, 5, 6]]]
        nose_poly_coll = Poly3DCollection(nose_tri, color="red", alpha=0.8, zorder=7)
        ax.add_collection3d(nose_poly_coll)

        # Remove old quivers
        for q_artist in (quiv_vel, quiv_acc, quiv_mag):
            if q_artist is not None:
                q_artist.remove()

        # Velocity arrow (green)
        v = vel_vec[idx] * VECTOR_SCALE["vel"]
        quiv_vel = ax.quiver(cx, cy, cz, v[0], v[1], v[2],
                             color="green", arrow_length_ratio=0.15, linewidth=1.8, label="Velocity")

        # Acceleration arrow (red)
        a = acc_vec[idx] * VECTOR_SCALE["acc"]
        quiv_acc = ax.quiver(cx, cy, cz, a[0], a[1], a[2],
                             color="red", arrow_length_ratio=0.15, linewidth=1.8, label="Acceleration")

        # Magnetic field arrow (blue)
        m = mag_vec[idx] * VECTOR_SCALE["mag"]
        quiv_mag = ax.quiver(cx, cy, cz, m[0], m[1], m[2],
                             color="dodgerblue", arrow_length_ratio=0.15, linewidth=1.8, label="Mag. bearing")

        # Info text
        vel_mag = np.linalg.norm(vel_vec[idx])
        acc_mag = np.linalg.norm(acc_vec[idx])
        info_text.set_text(
            f"Frame #{fp.frame_id:04d}  {fp.timestamp.strftime('%H:%M:%S.%f')[:-3]}\n"
            f"N={p.position.north_m - frames[0].pose.position.north_m:+.2f}m  "
            f"E={p.position.east_m - frames[0].pose.position.east_m:+.2f}m  "
            f"Alt={p.position.altitude_m:.2f}m\n"
            f"Yaw={p.orientation.yaw_deg:.1f}\u00b0  "
            f"Pitch={p.orientation.pitch_deg:.1f}\u00b0  "
            f"Roll={p.orientation.roll_deg:.1f}\u00b0\n"
            f"|V|={vel_mag:.2f} m/s  |A|={acc_mag:.2f} m/s\u00b2"
        )
        fig.canvas.draw_idle()

    # ---------- slider ----------
    ax_slider = fig.add_axes([0.15, 0.06, 0.55, 0.03])
    slider = Slider(ax_slider, "Frame", 0, len(frames) - 1, valinit=0, valstep=1, valfmt="%d")
    slider.on_changed(update)

    # ---------- prev / next buttons ----------
    ax_prev = fig.add_axes([0.75, 0.06, 0.06, 0.03])
    ax_next = fig.add_axes([0.82, 0.06, 0.06, 0.03])
    btn_prev = Button(ax_prev, "\u25c0 Prev")
    btn_next = Button(ax_next, "Next \u25b6")

    def on_prev(_event):
        slider.set_val(max(0, slider.val - 1))

    def on_next(_event):
        slider.set_val(min(len(frames) - 1, slider.val + 1))

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    # ---------- play / pause ----------
    ax_play = fig.add_axes([0.89, 0.06, 0.06, 0.03])
    btn_play = Button(ax_play, "\u25b6 Play")
    playback = {"active": False, "timer": None}

    def play_step():
        if not playback["active"]:
            return
        cur = int(slider.val)
        if cur < len(frames) - 1:
            slider.set_val(cur + 1)
            playback["timer"] = fig.canvas.new_timer(interval=50)
            playback["timer"].add_callback(play_step)
            playback["timer"].single_shot = True
            playback["timer"].start()
        else:
            playback["active"] = False
            btn_play.label.set_text("\u25b6 Play")

    def on_play(_event):
        if playback["active"]:
            playback["active"] = False
            btn_play.label.set_text("\u25b6 Play")
        else:
            playback["active"] = True
            btn_play.label.set_text("\u23f8 Stop")
            play_step()

    btn_play.on_clicked(on_play)

    # ---------- keyboard shortcuts ----------
    def on_key(event):
        if event.key == "left":
            on_prev(None)
        elif event.key == "right":
            on_next(None)
        elif event.key == " ":
            on_play(None)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ---------- legend ----------
    # Dummy artists for legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="green", linewidth=2, label="Velocity"),
        Line2D([0], [0], color="red", linewidth=2, label="Acceleration"),
        Line2D([0], [0], color="dodgerblue", linewidth=2, label="Mag. bearing"),
        Line2D([0], [0], color="steelblue", linewidth=1.5, label="Past trail"),
        Line2D([0], [0], color="0.70", linewidth=1, label="Full trail"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # Initial draw
    update(0)
    plt.show()


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else LOG_PATH
    frames = parse_telemetry_log(path)
    print(f"Loaded {len(frames)} frames from {path.name}")
    run(frames)


if __name__ == "__main__":
    main()
