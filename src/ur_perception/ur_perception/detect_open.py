"""Open-vocabulary detection service /detect_open (OWL-ViT through the transformers pipeline).

  ros2 run ur_perception detect_open
  ros2 run ur_perception detect_open --ros-args -p tiles:=3 -p threshold:=0.02
  ros2 service call /detect_open ur_interfaces/srv/DetectOpen "{labels: ['yellow can', 'purple ball']}"

Why tiles: a 5 cm object is only ~12 px wide in the 320x240 overhead image, too small for OWL-ViT at full-image
scale. The workspace is cropped (ROI) and cut into tiles x tiles overlapping squares; each is run separately so
objects look 2-3x bigger to the model. Boxes are merged with NMS.
On demand: the model loads on the first request. -p keep_loaded:=false releases VRAM after every request.
The arm must be at home (out of the camera's view). Pose z = TOP surface height, frame base_link.
Last camera image with boxes: /tmp/detect_open_last.png. Deps: scripts/install_detector.sh.
"""
import numpy as np, rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from ur_interfaces.srv import DetectOpen

CAM_POS = np.array([0.6, 0.0, 1.2])   # must match the world file / detect_objects.py
TABLE_TOP = 0.20                      # must match the world file
ABOVE_TABLE = 0.01                    # pixels must be this far above the table to count as object
MIN_PIX = 15                          # minimum object pixels inside a box
MIN_FILL = 0.25                       # object pixels / box area; the table box has ~0 and is rejected
MAX_BOX_FRAC = 0.30                   # of the ROI area; bigger boxes are the table, not an object
NMS_IOU = 0.5
ROI_X = (0.25, 0.95)                  # workspace to look at (base frame, metres)
ROI_Y = (-0.45, 0.45)
TILE_OVERLAP = 0.25
DEBUG_IMAGE = "/tmp/detect_open_last.png"


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets):
    """dets: list of (score, label, box). Keep the best box per object, across labels and tiles."""
    kept = []
    for d in sorted(dets, key=lambda d: -d[0]):
        if all(iou(d[2], k[2]) < NMS_IOU for k in kept):
            kept.append(d)
    return kept


def tile_boxes(roi, n):
    """Split roi=(x0,y0,x1,y1) into n x n overlapping tiles (n=1 -> the roi itself)."""
    x0, y0, x1, y1 = roi
    if n <= 1:
        return [roi]
    tw = (x1 - x0) / (1 + (n - 1) * (1 - TILE_OVERLAP))
    th = (y1 - y0) / (1 + (n - 1) * (1 - TILE_OVERLAP))
    out = []
    for j in range(n):
        for i in range(n):
            tx, ty = x0 + i * tw * (1 - TILE_OVERLAP), y0 + j * th * (1 - TILE_OVERLAP)
            out.append((tx, ty, tx + tw, ty + th))
    return out


class DetectOpenService(Node):
    def __init__(self):
        super().__init__("detect_open", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.model_name = self.declare_parameter("model", "google/owlvit-base-patch16").value
        self.default_thr = float(self.declare_parameter("threshold", 0.02).value)
        self.keep_loaded = self.declare_parameter("keep_loaded", True).value
        self.tiles = int(self.declare_parameter("tiles", 2).value)
        self.prompt = self.declare_parameter("prompt", "a photo of a {}").value
        self.br = CvBridge()
        self.rgb = self.depth = self.info = None
        self.pipe = None
        self.create_subscription(CameraInfo, "/overhead/camera_info", lambda m: setattr(self, "info", m), 1)
        self.create_subscription(Image, "/overhead/image", lambda m: setattr(self, "rgb", m), 1)
        self.create_subscription(Image, "/overhead/depth_image", lambda m: setattr(self, "depth", m), 1)
        self.create_service(DetectOpen, "/detect_open", self.on_request)
        self.get_logger().info(f"/detect_open ready (model {self.model_name}, tiles {self.tiles}; "
                               f"loads on the first request)")

    def _load(self):
        if self.pipe is None:
            import torch
            from transformers import pipeline
            dev = 0 if torch.cuda.is_available() else -1
            self.get_logger().info(f"loading {self.model_name} on {'GPU' if dev == 0 else 'CPU'} ...")
            self.pipe = pipeline("zero-shot-object-detection", model=self.model_name, device=dev)
            self.get_logger().info("model loaded")

    def _unload(self):
        self.pipe = None
        try:
            import torch, gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _roi(self, w, h):
        """Pixel rectangle of the workspace, from the camera model at table-top distance."""
        fx, fy, cx, cy = self.info.k[0], self.info.k[4], self.info.k[2], self.info.k[5]
        d = CAM_POS[2] - TABLE_TOP
        us = [cx - y * fx / d for y in ROI_Y]
        vs = [cy - (x - CAM_POS[0]) * fy / d for x in ROI_X]
        return (max(0.0, min(us)), max(0.0, min(vs)), min(float(w), max(us)), min(float(h), max(vs)))

    def _run(self, pil, tile_list, labels, thr, top_k=None):
        """Run the pipeline on every tile (padded to a square); return [(score, original_label, box_in_full_image)]."""
        from PIL import Image as PILImage
        prompted = {self.prompt.format(l): l for l in labels}
        out = []
        for (x0, y0, x1, y1) in tile_list:
            crop = pil.crop((int(x0), int(y0), int(x1), int(y1)))
            side = max(crop.size)
            sq = PILImage.new("RGB", (side, side), (128, 128, 128))
            sq.paste(crop, (0, 0))
            kw = {"top_k": top_k} if top_k else {}
            for d in self.pipe(sq, candidate_labels=list(prompted), threshold=thr, **kw):
                b = d["box"]
                out.append((float(d["score"]), prompted.get(d["label"], d["label"]),
                            (b["xmin"] + int(x0), b["ymin"] + int(y0), b["xmax"] + int(x0), b["ymax"] + int(y0))))
        return out

    def on_request(self, req, res):
        try:
            return self._handle(req, res)
        finally:
            if not self.keep_loaded:
                self._unload()

    def _handle(self, req, res):
        labels = [l.strip() for l in req.labels if l.strip()]
        if not labels:
            res.success, res.message = False, "give at least one label"
            return res
        if self.rgb is None or self.depth is None or self.info is None:
            res.success, res.message = False, "no camera data yet"
            return res
        try:
            self._load()
        except Exception as e:
            res.success = False
            res.message = f"could not load the detector ({e}). Run scripts/install_detector.sh"
            return res
        thr = req.threshold if req.threshold > 0 else self.default_thr
        try:
            from PIL import Image as PILImage, ImageDraw
            rgb = self.br.imgmsg_to_cv2(self.rgb, "rgb8")
            depth = self.br.imgmsg_to_cv2(self.depth, "32FC1")
            pil = PILImage.fromarray(rgb)
            h, w = rgb.shape[:2]
            roi = self._roi(w, h)
            tl = tile_boxes(roi, self.tiles)
            raw = self._run(pil, tl, labels, thr)
            floor = [] if raw else sorted(self._run(pil, tl, labels, 0.0, top_k=3), key=lambda d: -d[0])[:5]
        except Exception as e:
            res.success, res.message = False, f"detector failed: {e}"
            return res

        fin = depth[np.isfinite(depth) & (depth > 0)]
        dstat = (f"depth min/med/max {fin.min():.2f}/{np.median(fin):.2f}/{fin.max():.2f} m"
                 if fin.size else "depth image has no valid pixels")
        try:   # debug picture: ROI (cyan), tiles (grey), boxes (yellow)
            dbg = pil.copy()
            dr = ImageDraw.Draw(dbg)
            dr.rectangle(list(roi), outline=(0, 255, 255))
            for t in tl:
                dr.rectangle(list(t), outline=(120, 120, 120))
            for s, l, b in (raw or floor):
                dr.rectangle(list(b), outline=(255, 255, 0))
                dr.text((b[0] + 2, b[1] + 2), f"{l} {s:.2f}", fill=(255, 255, 255))
            dbg.save(DEBUG_IMAGE)
        except Exception:
            pass
        if not raw:
            best = "; ".join(f"{l} {s:.3f}" for s, l, _ in floor) or "no boxes at all"
            res.success = False
            res.message = (f"model: nothing above threshold {thr:.2f}. Best scores: {best}. {dstat}. "
                           f"Image with boxes: {DEBUG_IMAGE}")
            self.get_logger().warn(res.message)
            return res

        roi_area = (roi[2] - roi[0]) * (roi[3] - roi[1])
        dets, too_big = [], 0
        for s, l, b in raw:
            if (b[2] - b[0]) * (b[3] - b[1]) > MAX_BOX_FRAC * roi_area:
                too_big += 1
                continue
            dets.append((s, l, b))
        kept = nms(dets)

        fx, fy, cx, cy = self.info.k[0], self.info.k[4], self.info.k[2], self.info.k[5]
        counts, rejected = {}, 0
        for score, label, box in kept:
            x0, y0 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
            x1, y1 = min(w, int(round(box[2]))), min(h, int(round(box[3])))
            if x1 <= x0 or y1 <= y0:
                rejected += 1
                continue
            d = depth[y0:y1, x0:x1].astype(np.float64)
            vs, us = np.mgrid[y0:y1, x0:x1]
            ok = np.isfinite(d) & (d > 0) & ((CAM_POS[2] - d) > TABLE_TOP + ABOVE_TABLE)
            if ok.sum() < MIN_PIX or ok.sum() < MIN_FILL * d.size:
                rejected += 1          # empty or mostly table: not an object
                continue
            dd, u, v = d[ok], us[ok], vs[ok]
            xw = float(np.mean(-(v - cy) * dd / fy + CAM_POS[0]))
            yw = float(np.mean(-(u - cx) * dd / fx + CAM_POS[1]))
            top = float(CAM_POS[2] - np.percentile(dd, 20))
            ps = PoseStamped()
            ps.header.frame_id = "base_link"
            ps.header.stamp = self.depth.header.stamp
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = xw, yw, top
            ps.pose.orientation.w = 1.0
            k = counts.get(label, 0)
            counts[label] = k + 1
            res.ids.append(f"{label.replace(' ', '_')}_{k}")
            res.found_labels.append(label)
            res.scores.append(score)
            res.poses.append(ps)
        res.success = len(res.poses) > 0
        note = ""
        missing = [l for l in labels if l not in counts]
        if missing and res.success:      # explain every label that produced nothing
            parts = []
            for l in missing:
                try:
                    cand = sorted(self._run(pil, tl, [l], 0.0, top_k=1), key=lambda d: -d[0])
                except Exception:
                    cand = []
                if not cand:
                    parts.append(f"{l} (no box)")
                    continue
                sc, _, bx = cand[0]
                by = next((k[1] for k in kept if iou(bx, k[2]) >= NMS_IOU), None)
                parts.append(f"{l} (best {sc:.3f}" + (f", merged into stronger '{by}' box" if by else "") + ")")
            note = " | not found: " + "; ".join(parts)
        res.message = ((f"found {len(res.poses)} object(s)" + note) if res.success else
                       f"model gave {len(raw)} box(es): {too_big} too big, {len(dets) - len(kept)} merged by NMS, "
                       f"{rejected} rejected (too few object pixels / mostly table). {dstat}. "
                       f"Image with boxes: {DEBUG_IMAGE}")
        self.get_logger().info(f"{res.message} (raw={len(raw)}, kept={len(res.poses)})")
        return res


def main():
    rclpy.init()
    rclpy.spin(DetectOpenService())
