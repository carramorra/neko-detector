# =================================================================
# MEOWMIRROR PRO v5.0 — FULL 16-CAT DETECTION ENGINE
# =================================================================
# pip install opencv-python mediapipe numpy
#
# CAT ASSET FILENAMES EXPECTED (in assets/ folder):
#   actually.jpeg   cat-tongue.jpeg  cooked.jpeg    cute-hand.jpeg
#   explode.jpeg    gimme.jpeg       glare.jpeg     jawline.jpeg
#   lol.jpeg        punch.jpeg       rejecting.jpeg shock.jpeg
#   side-eye.jpeg   squint.jpeg      stress.jpeg    thumbsup.jpeg
#
# SIGNAL → CAT MAPPING:
#   FACE EXPRESSIONS:
#     SHOCK      → shock.jpeg       (mouth open wide + big eyes)
#     LOL        → lol.jpeg         (huge mouth open + wide)
#     COOKED     → cooked.jpeg      (derp eyes wide + brows up)
#     SQUINT     → squint.jpeg      (eyes nearly closed)
#     GLARE      → glare.jpeg       (squint + low angry brows)
#     STRESS     → stress.jpeg      (tight mouth + low brows)
#     SIDE_EYE   → side-eye.jpeg    (head tilted or eye asymmetry)
#     JAWLINE    → jawline.jpeg     (chin raised, mouth closed)
#     CAT_TONGUE → cat-tongue.jpeg  (small mouth pout open)
#     EXPLODE    → explode.jpeg     (brows max high + wide eyes + open mouth)
#
#   HAND GESTURES:
#     THUMBSUP   → thumbsup.jpeg    (thumb up fist)
#     ACTUALLY   → actually.jpeg    (index finger only pointing up)
#     GIMME      → gimme.jpeg       (all fingers curled / grabbing)
#     REJECTING  → rejecting.jpeg   (open palm stop gesture)
#     PUNCH      → punch.jpeg       (closed fist)
#     CUTE_HAND  → cute-hand.jpeg   (pinky + index up / horns)
# =================================================================

import cv2
import mediapipe as mp
import os
import math
import numpy as np
from collections import deque

# ─── Face Landmark Indices ────────────────────────────────────────
NOSE_TIP       = 1
LEFT_EYE_TOP   = 159;  LEFT_EYE_BOT    = 145
RIGHT_EYE_TOP  = 386;  RIGHT_EYE_BOT   = 374
LEFT_BROW_IN   = 107;  LEFT_EYE_IN     = 133
RIGHT_BROW_IN  = 336;  RIGHT_EYE_OUT   = 263
LEFT_BROW_MID  = 105;  RIGHT_BROW_MID  = 334
MOUTH_TOP      = 13;   MOUTH_BOT       = 14
MOUTH_LEFT     = 61;   MOUTH_RIGHT     = 291
LEFT_CHEEK     = 234;  RIGHT_CHEEK     = 454
CHIN           = 152;  FOREHEAD        = 10

# ─── Hand Landmark Indices ────────────────────────────────────────
WRIST  = 0
TH_TIP = 4;  TH_IP  = 3;  TH_MCP = 2;  TH_CMC = 1
IX_TIP = 8;  IX_PIP = 6;  IX_MCP = 5
MI_TIP = 12; MI_PIP = 10; MI_MCP = 9
RI_TIP = 16; RI_PIP = 14; RI_MCP = 13
PI_TIP = 20; PI_PIP = 18; PI_MCP = 17

# ─── Signal → Asset filename ─────────────────────────────────────
SIGNAL_TO_ASSET = {
    "SHOCK":      "shock",
    "LOL":        "lol",
    "COOKED":     "cooked",
    "SQUINT":     "squint",
    "GLARE":      "glare",
    "STRESS":     "stress",
    "SIDE_EYE":   "side-eye",
    "JAWLINE":    "jawline",
    "CAT_TONGUE": "cat-tongue",
    "EXPLODE":    "explode",
    "THUMBSUP":   "thumbsup",
    "ACTUALLY":   "actually",
    "GIMME":      "gimme",
    "REJECTING":  "rejecting",
    "PUNCH":      "punch",
    "CUTE_HAND":  "cute-hand",
}

# ─── HUD label shown on screen ───────────────────────────────────
SIGNAL_LABEL = {
    "SHOCK":      "SHOCK",
    "LOL":        "LOL",
    "COOKED":     "COOKED",
    "SQUINT":     "SQUINT",
    "GLARE":      "GLARE",
    "STRESS":     "STRESS",
    "SIDE_EYE":   "SIDE-EYE",
    "JAWLINE":    "JAWLINE",
    "CAT_TONGUE": "CAT-TONGUE",
    "EXPLODE":    "EXPLODE",
    "THUMBSUP":   "THUMBS UP",
    "ACTUALLY":   "ACTUALLY",
    "GIMME":      "GIMME",
    "REJECTING":  "REJECTING",
    "PUNCH":      "PUNCH",
    "CUTE_HAND":  "CUTE-HAND",
    "NEUTRAL":    "NEUTRAL",
}


# ═════════════════════════════════════════════════════════════════
#  SIGNAL SMOOTHER  — prevents flickering
# ═════════════════════════════════════════════════════════════════
class SignalSmoother:
    def __init__(self, window=10, threshold=6):
        self.history  = deque(maxlen=window)
        self.threshold = threshold
        self.stable    = "NEUTRAL"

    def update(self, raw):
        self.history.append(raw)
        if self.history.count(raw) >= self.threshold:
            self.stable = raw
        return self.stable


# ═════════════════════════════════════════════════════════════════
#  MAIN ENGINE
# ═════════════════════════════════════════════════════════════════
class MeowMirrorPro:

    # ── Tracker palette ──────────────────────────────────────────
    C_FACE_TESS   = (25,  55,  75)
    C_FACE_CONT   = (0,   140, 200)
    C_IRIS        = (0,   230, 180)
    C_FACE_DOT    = (0,   210, 255)
    C_HAND_CONN   = (0,   190, 90)
    C_TIP_OUTER   = (0,   255, 110)
    C_TIP_INNER   = (255, 255, 255)
    C_JOINT       = (0,   200, 130)
    C_WRIST       = (0,   160, 255)
    C_HUD_ACCENT  = (0,   200, 255)
    C_CAT_BORDER1 = (0,   100, 160)
    C_CAT_BORDER2 = (0,   220, 255)

    def __init__(self, assets_dir="assets"):
        self.assets_dir = assets_dir

        mp_h = mp.solutions.hands
        self.mp_hands = mp_h
        self.hands = mp_h.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.65,
            model_complexity=1,
        )

        mp_f = mp.solutions.face_mesh
        self.mp_face = mp_f
        self.face_mesh = mp_f.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,          # enables iris tracking
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )

        self.mp_draw  = mp.solutions.drawing_utils
        self.smoother = SignalSmoother(window=10, threshold=6)
        self._cache   = {}

    # ─────────────────────────────────────────────────────────────
    #  UTILS
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _d(a, b):
        return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

    def _asset(self, name):
        if name not in self._cache:
            for ext in ("jpeg", "jpg", "png"):
                p = os.path.join(self.assets_dir, f"{name}.{ext}")
                if os.path.exists(p):
                    self._cache[name] = cv2.imread(p)
                    break
            else:
                self._cache[name] = None
        return self._cache[name]

    # ─────────────────────────────────────────────────────────────
    #  FACE EXPRESSION ANALYSIS
    # ─────────────────────────────────────────────────────────────
    def _face_signal(self, lm):
        face_h = self._d(lm[FOREHEAD], lm[CHIN]) + 1e-6

        mouth_open = self._d(lm[MOUTH_TOP],  lm[MOUTH_BOT])  / face_h
        mouth_wide = self._d(lm[MOUTH_LEFT], lm[MOUTH_RIGHT]) / face_h
        l_eye_h    = self._d(lm[LEFT_EYE_TOP],  lm[LEFT_EYE_BOT])  / face_h
        r_eye_h    = self._d(lm[RIGHT_EYE_TOP], lm[RIGHT_EYE_BOT]) / face_h
        avg_eye    = (l_eye_h + r_eye_h) / 2

        l_brow     = self._d(lm[LEFT_BROW_IN],  lm[LEFT_EYE_IN])   / face_h
        r_brow     = self._d(lm[RIGHT_BROW_IN], lm[RIGHT_EYE_OUT]) / face_h
        avg_brow   = (l_brow + r_brow) / 2

        tilt       = abs(lm[LEFT_CHEEK].y - lm[RIGHT_CHEEK].y) / face_h
        eye_asym   = abs(l_eye_h - r_eye_h) / (avg_eye + 1e-6)
        jaw_ratio  = (lm[CHIN].y - lm[NOSE_TIP].y) / face_h

        scores = {}

        # EXPLODE: brows sky-high + wide eyes + open mouth
        if avg_brow > 0.10 and avg_eye > 0.028 and mouth_open > 0.04:
            scores["EXPLODE"] = avg_brow*6 + avg_eye*5 + mouth_open*4

        # LOL: mouth very wide AND very open
        if mouth_open > 0.065 and mouth_wide > 0.22:
            scores["LOL"] = mouth_wide*8 + mouth_open*5

        # SHOCK: mouth very open, eyes wide
        if mouth_open > 0.055 and avg_eye > 0.026:
            scores["SHOCK"] = mouth_open*9 + avg_eye*4

        # COOKED: huge eyes + raised brows (derp face)
        if avg_eye > 0.032 and avg_brow > 0.09:
            scores["COOKED"] = avg_eye*8 + avg_brow*4

        # CAT_TONGUE: small mouth opening only
        if 0.022 < mouth_open < 0.050 and mouth_wide < 0.19:
            scores["CAT_TONGUE"] = mouth_open * 7

        # GLARE: narrow eyes + low brows
        if avg_eye < 0.015 and avg_brow < 0.060:
            scores["GLARE"] = (0.015-avg_eye)*35 + (0.060-avg_brow)*20

        # SQUINT: narrow eyes, brows neutral
        if avg_eye < 0.013 and avg_brow >= 0.060:
            scores["SQUINT"] = (0.013-avg_eye)*40

        # STRESS: tight mouth + low brows
        if avg_brow < 0.058 and mouth_open < 0.022 and mouth_wide > 0.17:
            scores["STRESS"] = (0.058-avg_brow)*18 + mouth_wide*5

        # SIDE_EYE: head tilted OR asymmetric eye openness
        if tilt > 0.030 or eye_asym > 0.28:
            scores["SIDE_EYE"] = tilt*16 + eye_asym*5

        # JAWLINE: chin raised, mouth closed
        if jaw_ratio < 0.34 and mouth_open < 0.025:
            scores["JAWLINE"] = (0.34 - jaw_ratio) * 22

        if not scores:
            return "NEUTRAL"
        return max(scores, key=scores.get)

    # ─────────────────────────────────────────────────────────────
    #  HAND GESTURE ANALYSIS
    # ─────────────────────────────────────────────────────────────
    def _ext(self, lm, tip, pip):
        return lm[tip].y < lm[pip].y

    def _thumb_up_check(self, lm):
        return (lm[TH_TIP].y < lm[TH_IP].y < lm[TH_MCP].y)

    def _hand_signal(self, lm):
        ix = self._ext(lm, IX_TIP, IX_PIP)
        mi = self._ext(lm, MI_TIP, MI_PIP)
        ri = self._ext(lm, RI_TIP, RI_PIP)
        pi = self._ext(lm, PI_TIP, PI_PIP)
        th = self._thumb_up_check(lm)

        # THUMBSUP — only thumb pointing up, others curled
        if th and not ix and not mi and not ri and not pi:
            return "THUMBSUP"

        # ACTUALLY — only index finger pointing up
        if ix and not mi and not ri and not pi and not th:
            return "ACTUALLY"

        # CUTE_HAND — index + pinky up (horns / rock-on)
        if ix and pi and not mi and not ri:
            return "CUTE_HAND"

        # PUNCH — full closed fist
        if not ix and not mi and not ri and not pi and not th:
            return "PUNCH"

        # REJECTING — all 4 fingers extended (open palm / stop)
        if ix and mi and ri and pi:
            return "REJECTING"

        # GIMME — middle + ring + pinky extended, index curled
        if mi and ri and pi and not ix:
            return "GIMME"

        return None

    # ─────────────────────────────────────────────────────────────
    #  FULL DETECTION
    # ─────────────────────────────────────────────────────────────
    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_res = self.hands.process(rgb)
        face_res = self.face_mesh.process(rgb)
        return hand_res, face_res

    def get_signal(self, hand_res, face_res):
        raw = "NEUTRAL"

        if face_res.multi_face_landmarks:
            lm  = face_res.multi_face_landmarks[0].landmark
            raw = self._face_signal(lm)

        # Hand overrides face
        if hand_res.multi_hand_landmarks:
            for hlm in hand_res.multi_hand_landmarks:
                g = self._hand_signal(hlm.landmark)
                if g:
                    raw = g
                    break

        return self.smoother.update(raw)

    # ─────────────────────────────────────────────────────────────
    #  HIGH-QUALITY FACE TRACKER
    # ─────────────────────────────────────────────────────────────
    def draw_face_tracker(self, frame, face_res, h, w):
        if not face_res.multi_face_landmarks:
            return frame

        for face_lm in face_res.multi_face_landmarks:
            overlay = frame.copy()

            # 1. Fine tessellation (468-pt mesh)
            self.mp_draw.draw_landmarks(
                overlay, face_lm,
                self.mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_draw.DrawingSpec(
                    color=self.C_FACE_TESS, thickness=1, circle_radius=0)
            )
            # 2. Bold contour outline
            self.mp_draw.draw_landmarks(
                overlay, face_lm,
                self.mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_draw.DrawingSpec(
                    color=self.C_FACE_CONT, thickness=1, circle_radius=0)
            )
            # 3. Iris rings
            self.mp_draw.draw_landmarks(
                overlay, face_lm,
                self.mp_face.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_draw.DrawingSpec(
                    color=self.C_IRIS, thickness=1, circle_radius=0)
            )

            # Blend mesh so it doesn't overpower the face
            cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

            # 4. Key landmark glowing dots (drawn after blend)
            KEY_PTS = [
                NOSE_TIP,
                LEFT_EYE_TOP, LEFT_EYE_BOT, RIGHT_EYE_TOP, RIGHT_EYE_BOT,
                LEFT_BROW_IN, RIGHT_BROW_IN, LEFT_BROW_MID, RIGHT_BROW_MID,
                MOUTH_TOP, MOUTH_BOT, MOUTH_LEFT, MOUTH_RIGHT,
                LEFT_CHEEK, RIGHT_CHEEK, CHIN, FOREHEAD,
            ]
            for idx in KEY_PTS:
                pt = face_lm.landmark[idx]
                px, py = int(pt.x * w), int(pt.y * h)
                # soft outer glow
                cv2.circle(frame, (px, py), 7, self.C_FACE_DOT, 2)
                # bright filled core
                cv2.circle(frame, (px, py), 3, self.C_FACE_DOT, -1)
                # crisp white ring
                cv2.circle(frame, (px, py), 3, (255, 255, 255), 1)

        return frame

    # ─────────────────────────────────────────────────────────────
    #  HIGH-QUALITY HAND TRACKER
    # ─────────────────────────────────────────────────────────────
    def draw_hand_tracker(self, frame, hand_res, h, w):
        if not hand_res.multi_hand_landmarks:
            return frame

        TIPS = {TH_TIP, IX_TIP, MI_TIP, RI_TIP, PI_TIP}

        for hand_lm in hand_res.multi_hand_landmarks:
            overlay = frame.copy()

            # Bone connections
            self.mp_draw.draw_landmarks(
                overlay, hand_lm,
                self.mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(0,0,0), thickness=0, circle_radius=0),
                connection_drawing_spec=self.mp_draw.DrawingSpec(
                    color=self.C_HAND_CONN, thickness=2, circle_radius=0)
            )
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            # Custom dots
            for i, pt in enumerate(hand_lm.landmark):
                px, py = int(pt.x * w), int(pt.y * h)

                if i in TIPS:
                    # Fingertips: large triple-layer dot
                    cv2.circle(frame, (px, py), 11, self.C_TIP_OUTER, -1)
                    cv2.circle(frame, (px, py),  7, (0, 180, 80),     -1)
                    cv2.circle(frame, (px, py),  4, self.C_TIP_INNER, -1)
                elif i == WRIST:
                    # Wrist: blue anchor dot
                    cv2.circle(frame, (px, py),  9, self.C_WRIST,     -1)
                    cv2.circle(frame, (px, py),  5, (255, 255, 255),  -1)
                else:
                    # Mid-joints: smaller green dot
                    cv2.circle(frame, (px, py),  6, self.C_JOINT,     -1)
                    cv2.circle(frame, (px, py),  3, (255, 255, 255),  -1)

        return frame

    # ─────────────────────────────────────────────────────────────
    #  HUD OVERLAY
    # ─────────────────────────────────────────────────────────────
    def draw_hud(self, frame, signal, h, w):
        label = SIGNAL_LABEL.get(signal, signal)

        # Top dark bar
        bar = np.zeros((76, w, 3), dtype=np.uint8)
        bar[:] = (12, 12, 18)
        cv2.rectangle(bar, (0, 0), (5, 76), self.C_HUD_ACCENT, -1)
        cv2.putText(bar, "MEOWMIRROR PRO  v5.0",
                    (16, 24), cv2.FONT_HERSHEY_DUPLEX, 0.52, (70, 70, 95), 1)
        sig_clr = (0, 255, 120) if signal != "NEUTRAL" else (90, 90, 90)
        cv2.putText(bar, f">> {label}",
                    (16, 60), cv2.FONT_HERSHEY_DUPLEX, 0.90, sig_clr, 2)
        frame[0:76, :] = bar

        # Corner scan brackets
        m, arm, top = 18, 45, 82
        clr, th = (180, 210, 255), 2

        def corner(x1, y1, x2, y2):
            cv2.line(frame, (x1, y1), (x1+arm, y1), clr, th)
            cv2.line(frame, (x1, y1), (x1, y1+arm), clr, th)
            cv2.line(frame, (x2, y1), (x2-arm, y1), clr, th)
            cv2.line(frame, (x2, y1), (x2, y1+arm), clr, th)
            cv2.line(frame, (x1, y2), (x1+arm, y2), clr, th)
            cv2.line(frame, (x1, y2), (x1, y2-arm), clr, th)
            cv2.line(frame, (x2, y2), (x2-arm, y2), clr, th)
            cv2.line(frame, (x2, y2), (x2, y2-arm), clr, th)

        corner(m, top, w-m, h-m)
        return frame

    # ─────────────────────────────────────────────────────────────
    #  CAT PANEL (bottom-right)
    # ─────────────────────────────────────────────────────────────
    def draw_cat_panel(self, frame, signal, h, w):
        if signal == "NEUTRAL":
            return frame
        asset_name = SIGNAL_TO_ASSET.get(signal)
        if not asset_name:
            return frame
        img = self._asset(asset_name)
        if img is None:
            cv2.putText(frame, f"[{asset_name} not found]",
                        (w-280, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60,60,80), 1)
            return frame

        SIZE = 230
        try:
            cat = cv2.resize(img, (SIZE, SIZE))
            x1 = w - SIZE - 20;  y1 = h - SIZE - 20
            x2 = x1 + SIZE;      y2 = y1 + SIZE
            frame[y1-6:y2+6, x1-6:x2+6] = (10, 10, 16)
            frame[y1:y2, x1:x2] = cat
            cv2.rectangle(frame, (x1-4, y1-4), (x2+4, y2+4), self.C_CAT_BORDER1, 3)
            cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), self.C_CAT_BORDER2, 1)
            cv2.putText(frame, asset_name.upper(),
                        (x1, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.52, self.C_CAT_BORDER2, 1)
        except Exception:
            pass
        return frame

    # ─────────────────────────────────────────────────────────────
    #  LEGEND (press L to toggle)
    # ─────────────────────────────────────────────────────────────
    def draw_legend(self, frame, h, w):
        lines = [
            "--- FACE EXPRESSIONS ---",
            "Open mouth wide    -> SHOCK",
            "Open + wide laugh  -> LOL",
            "Wide eyes + brows  -> COOKED",
            "Slight mouth open  -> CAT-TONGUE",
            "Narrow eyes only   -> SQUINT",
            "Squint + low brow  -> GLARE",
            "Tight mouth        -> STRESS",
            "Head tilt / asym   -> SIDE-EYE",
            "Chin raised        -> JAWLINE",
            "Brows+eyes+mouth   -> EXPLODE",
            "",
            "--- HAND GESTURES ---",
            "Thumb up, fist     -> THUMBSUP",
            "Index only up      -> ACTUALLY",
            "Pinky+index up     -> CUTE-HAND",
            "Closed fist        -> PUNCH",
            "Open palm          -> REJECTING",
            "Mid+ring+pinky up  -> GIMME",
        ]
        pad, lh = 10, 19
        panel_w, panel_h = 295, len(lines)*lh + pad*2
        x0, y0 = 10, h - panel_h - 10
        ov = frame.copy()
        cv2.rectangle(ov, (x0, y0), (x0+panel_w, y0+panel_h), (10, 10, 22), -1)
        cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0+panel_w, y0+panel_h), self.C_HUD_ACCENT, 1)
        for i, line in enumerate(lines):
            clr = (0, 200, 255) if line.startswith("---") else (200, 225, 200)
            cv2.putText(frame, line, (x0+pad, y0+pad+lh*(i+1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1)
        return frame


# ═════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═════════════════════════════════════════════════════════════════
def main():
    engine = MeowMirrorPro(assets_dir="assets")
    cap    = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    signal      = "NEUTRAL"
    show_legend = False

    print("=" * 56)
    print("  MeowMirror Pro v5.0  —  16-Cat Detection Engine")
    print("=" * 56)
    print("  Q = quit   |   L = toggle signal legend")
    print("=" * 56)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Detect
        hand_res, face_res = engine.detect(frame)
        signal = engine.get_signal(hand_res, face_res)

        # Tracker layers
        frame = engine.draw_face_tracker(frame, face_res, h, w)
        frame = engine.draw_hand_tracker(frame, hand_res, h, w)

        # UI layers
        frame = engine.draw_hud(frame, signal, h, w)
        frame = engine.draw_cat_panel(frame, signal, h, w)

        if show_legend:
            frame = engine.draw_legend(frame, h, w)

        cv2.imshow("MeowMirror Pro v5.0", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            show_legend = not show_legend

    cap.release()
    cv2.destroyAllWindows()
    print("MeowMirror closed.")


if __name__ == "__main__":
    main()