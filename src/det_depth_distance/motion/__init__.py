"""Per-track 2D motion estimation and trajectory prediction (numpy only).

Optional stage after distance/smoothing (YAML section `motion`; not to be
confused with `tracking.motion`, the tracker's box prediction). For every
track it measures the object's position on the plane from its box reference
point, its distance and the depth-map intrinsics, filters it with a motion
model chosen by the track's canonical class at birth, and predicts the
trajectory with its covariance.

Frames and conventions:
- camera: x right, y down, z along the optical axis (m); the state uses (x, z).
- heading psi: velocity = |v| (sin psi, cos psi) in (x, z); 0 = moving away,
  +90 deg = moving right; yaw rate > 0 = turning right (seen from above).
- world (only with ego-motion): REP-103 2D, x/y in metres, yaw CCW (rad);
  vehicle: x forward, y left; camera_mount places the camera in the vehicle.
- Without ego-motion everything is relative to the camera: the own
  vehicle's rotation shows up as apparent lateral motion of the objects.
  With it, (vx, vz) stay the camera-relative velocity, while the heading
  (and min_speed_for_heading) and the yaw rate describe the motion over
  ground, the heading expressed in the current camera axes.
All times are seconds from Frame.t (stride-aware), never frame counts.
Depth noise is temporally correlated (AR(1), rho ~0.7): filtering removes
jitter, not bias.

Import rule: depends on core.types, core.registry, distance and temporal;
config and core.pipeline import from here, never the reverse.
"""

from det_depth_distance.motion.config import MotionConfig
from det_depth_distance.motion.estimator import MotionEstimator, build_motion

__all__ = ["MotionConfig", "MotionEstimator", "build_motion"]
