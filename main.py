import os
import subprocess  # For playing sound effect
import time  # For the countdown timer
from datetime import datetime  # For the name and tagging of photos

import cv2
import mediapipe as mp
import numpy as np

import sys
import time
import threading
import itertools

class Spinner:
    def __init__(self, message="Loading AI Models..."):
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.delay = 0.1
        self.message = message
        self.running = False
        self.thread = None

    def spin(self):
        while self.running:
            sys.stdout.write(f"\r\033[36m{next(self.spinner)}\033[0m {self.message}")
            sys.stdout.flush()
            time.sleep(self.delay)
            
        sys.stdout.write(f"\r\033[32m✔\033[0m {self.message} Done!          \n")
        sys.stdout.flush()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

class VirtualPhotobooth:
    # Main photobooth class handling face detection and prop overlay

    def __init__(self):
        # init mediapipe, Rembg and assets
        # Loads all dynamic assets (backgrounds from the folder, transparent PNG props)
        # and sets up default toggle states.

        # face mesh setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # mediaPipe for live preview
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )

        # rembg for final render
        try:
            print("Loading...")
            from rembg import new_session

            # print("\nLoading HD Human Segmentation Model (This takes a moment)...\n")
            # Using 'u2net' to support multiple people!
            self.rembg_session = new_session("u2net")
        except ImportError:
            print("WARNING: 'rembg' is not installed. Final photos will use MediaPipe.")
            self.rembg_session = None

        # load backgrounds
        self.bg_folder = "assets/backgrounds"
        self.bg_images = []
        self.current_bg_idx = 0

        if not os.path.exists(self.bg_folder):
            os.makedirs(self.bg_folder)
            print(
                f"\nCreated folder: '{self.bg_folder}'. Drop your background images here!\n"
            )

        print("\nLoading Virtual Backgrounds...")
        for filename in sorted(os.listdir(self.bg_folder)):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(self.bg_folder, filename)
                img = cv2.imread(path)
                if img is not None:
                    self.bg_images.append(img)
                    print(f"  -> Loaded BG: {filename}")

        if not self.bg_images:
            print(
                f"  -> WARNING: No images found in {self.bg_folder}. No backgrounds available at all!"
            )

        self.output_dir = "graduation_photos"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print("\nLoading custom transparent assets...")

        def load_png(filename):
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"  -> WARNING: Could not find {filename}")
            return img

        self.assets = {
            "beard": load_png("assets/Beard.png"),
            "cap": load_png("assets/Graduation Cap.png"),
            "mask": load_png("assets/Mask.png"),
            "propeller": load_png("assets/Propeller Hat.png"),
            "sst_glasses": load_png("assets/SST Glasses.png"),
            "sunglasses": load_png("assets/Sunglasses.png"),
            "heartglasses": load_png("assets/heart2.png"),
        }

        self.active_props = {
            "hat": "cap",
            "eyes": "sst_glasses",
            "face": "mask",
            "banner": True,
            "confetti": True,
            "frame": True,
            "virtual_bg": False,
            "squad_sparks": True,
            "halo_streaks": False,
            "wireframe": False,
        }

        # load logo
        self.sst_logo = load_png("assets/logo/sst_logo.png")
        if self.sst_logo is not None:
            # if image doesn't have an alpha channel, add a solid one so overlay logic doesn't crash
            if len(self.sst_logo.shape) < 3 or self.sst_logo.shape[2] == 3:
                self.sst_logo = cv2.cvtColor(self.sst_logo, cv2.COLOR_BGR2BGRA)

            # resize height to 60px to fit inside the 80px banner while keeping aspect ratio
            h, w = self.sst_logo.shape[:2]
            target_h = 60
            target_w = int(w * (target_h / h))
            self.sst_logo = cv2.resize(self.sst_logo, (target_w, target_h))

        self.accumulated_frame = None

        self.is_counting_down = False
        self.countdown_start_time = 0

        self.sst_blue = (139, 69, 19)
        self.gold = (0, 215, 255)
        self.white = (255, 255, 255)

    def play_shutter_sound(self):
        # Triggers the camera shutter sound effect (afplay) natively on macOS without freezing the camera feed.
        # So non-block essentially
        sound_file = "assets/shutter.mp3"
        if not os.path.exists(sound_file):
            print("Audio file not found")
            return

        try:
            subprocess.Popen(["afplay", sound_file])
        except Exception as e:
            print(f"Audio playback skipped: {e}")

    def apply_virtual_background(self, frame):
        # use MediaPipe's lightweight AI to swap the background at relatively high FPS. It is fast but has coarse edges.
        if not self.bg_images:
            return frame

        bg_image = self.bg_images[self.current_bg_idx]
        h, w, _ = frame.shape
        bg_resized = cv2.resize(bg_image, (w, h))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = self.segmentation.process(frame_rgb).segmentation_mask
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2)
        mask_3d = np.stack((mask,) * 3, axis=-1)
        foreground = frame.astype(float) * mask_3d
        background = bg_resized.astype(float) * (1.0 - mask_3d)
        return (foreground + background).astype(np.uint8)

    def apply_high_quality_background(self, frame):
        # Use the rembg (u2net) model for final saved photo
        if not self.bg_images or self.rembg_session is None:
            return self.apply_virtual_background(frame)

        from rembg import remove

        bg_image = self.bg_images[self.current_bg_idx]
        h, w, _ = frame.shape
        bg_resized = cv2.resize(bg_image, (w, h))

        result_rgba = remove(frame, session=self.rembg_session, post_process_mask=True)
        fg = result_rgba[:, :, :3]
        alpha = result_rgba[:, :, 3]

        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
        alpha_float = alpha / 255.0
        alpha_3d = np.stack((alpha_float,) * 3, axis=-1)

        final_frame = fg.astype(float) + bg_resized.astype(float) * (1.0 - alpha_3d)
        return np.clip(final_frame, 0, 255).astype(np.uint8)

    def _overlay_image_alpha(
        self, img, img_overlay, x, y, angle=0.0, target_width=None
    ):
        # handles rotation/scaling for the png overlay
        if img_overlay is None:
            return img

        if target_width is not None:
            h, w = img_overlay.shape[:2]
            scale = target_width / w * 1.2
            img_overlay = cv2.resize(img_overlay, (int(w * scale), int(h * scale)))

        if angle != 0.0:
            h, w = img_overlay.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)

            cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            img_overlay = cv2.warpAffine(
                img_overlay,
                M,
                (new_w, new_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

        h, w = img_overlay.shape[:2]
        y1, y2 = int(y - h / 2), int(y - h / 2) + h
        x1, x2 = int(x - w / 2), int(x - w / 2) + w

        y1_clip, y2_clip = max(0, y1), min(img.shape[0], y2)
        x1_clip, x2_clip = max(0, x1), min(img.shape[1], x2)

        if y1_clip >= y2_clip or x1_clip >= x2_clip:
            return img

        oy1, oy2 = y1_clip - y1, y2_clip - y1
        ox1, ox2 = x1_clip - x1, x2_clip - x1

        alpha_mask = img_overlay[oy1:oy2, ox1:ox2, 3] / 255.0
        alpha_inv = 1.0 - alpha_mask
        for c in range(3):
            img[y1_clip:y2_clip, x1_clip:x2_clip, c] = (
                alpha_mask * img_overlay[oy1:oy2, ox1:ox2, c]
                + alpha_inv * img[y1_clip:y2_clip, x1_clip:x2_clip, c]
            )
        return img

    def _draw_star(self, image, center, size, color, fill_color, tilt_angle=0):
        # Draw a star
        points = []
        for i in range(10):
            base_angle = i * np.pi / 5 - np.pi / 2
            final_angle = base_angle + np.radians(tilt_angle)
            radius = size if i % 2 == 0 else size * 0.5
            x = int(center[0] + radius * np.cos(final_angle))
            y = int(center[1] + radius * np.sin(final_angle))
            points.append([x, y])
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(image, [points], fill_color)
        cv2.polylines(image, [points], True, color, 3)

    def draw_rounded_rect(
        self, img, top_left, bottom_right, color, radius=10, thickness=1, filled=False
    ):
        # Draws a rounded rectangle using OpenCV primitives. (this was a workaround as OpenCV does not have it)
        x1, y1 = top_left
        x2, y2 = bottom_right

        radius = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

        if filled:
            # Draw central intersecting rectangles
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            # Fill the four corners with circles
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
        else:
            # Draw four straight lines
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
            # Draw four corner arcs
            cv2.ellipse(
                img,
                (x1 + radius, y1 + radius),
                (radius, radius),
                180,
                0,
                90,
                color,
                thickness,
            )
            cv2.ellipse(
                img,
                (x2 - radius, y1 + radius),
                (radius, radius),
                270,
                0,
                90,
                color,
                thickness,
            )
            cv2.ellipse(
                img,
                (x2 - radius, y2 - radius),
                (radius, radius),
                0,
                0,
                90,
                color,
                thickness,
            )
            cv2.ellipse(
                img,
                (x1 + radius, y2 - radius),
                (radius, radius),
                90,
                0,
                90,
                color,
                thickness,
            )

    def draw_squad_connections(self, image, face_centers):
        # Calculates the distance between multiple people in the frame.
        # Draws a blue/gold line between them, adding a glowing star if they stand close together.
        num_faces = len(face_centers)
        if num_faces < 2:
            return
        for i in range(num_faces):
            pt1 = face_centers[i]
            pt2 = face_centers[(i + 1) % num_faces]
            distance = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
            thickness = max(2, int(15 - (distance / 40)))
            if distance < 400:
                color = self.gold
                mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                self._draw_star(
                    image, mid_point, int(thickness * 1.5), self.white, self.gold
                )
            else:
                color = self.sst_blue
            cv2.line(image, pt1, pt2, color, thickness)

    def draw_banner(self, image):
        # Renders the semi-transparent "SST Graduation Tea 2026" header at the top of the screen. (and the SST logo)
        height, width = image.shape[:2]
        banner_height = 80
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, banner_height), self.sst_blue, -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        cv2.rectangle(image, (0, 0), (width, banner_height), self.gold, 3)

        if hasattr(self, "sst_logo") and self.sst_logo is not None:
            logo_w = self.sst_logo.shape[1]
            logo_h = self.sst_logo.shape[0]
            logo_x = 30 + (logo_w // 2)
            logo_y = banner_height // 2

            # Calculate the boundaries of the badge with padding
            pad_x = 12
            pad_y = 3
            top_left = (logo_x - (logo_w // 2) - pad_x, logo_y - (logo_h // 2) - pad_y)
            bottom_right = (
                logo_x + (logo_w // 2) + pad_x,
                logo_y + (logo_h // 2) + pad_y,
            )

            self.draw_rounded_rect(
                image, top_left, bottom_right, self.white, radius=12, filled=True
            )

            self.draw_rounded_rect(
                image,
                top_left,
                bottom_right,
                self.gold,
                radius=12,
                thickness=2,
                filled=False,
            )

            self._overlay_image_alpha(image, self.sst_logo, logo_x, logo_y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "SST GRADUATION TEA 2026"
        text_size = cv2.getTextSize(text, font, 1.2, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (banner_height + text_size[1]) // 2

        cv2.putText(image, text, (text_x + 2, text_y + 2), font, 1.2, (0, 0, 0), 3)
        cv2.putText(image, text, (text_x, text_y), font, 1.2, self.gold, 3)

        bottom_text = "Celebrating 4 Years of Innovation"
        text_size2 = cv2.getTextSize(bottom_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x2 = (width - text_size2[0]) // 2
        cv2.putText(
            image,
            bottom_text,
            (text_x2, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.gold,
            2,
        )

    def draw_confetti(self, image, frame_count):
        # confetti animation!
        height, width = image.shape[:2]
        np.random.seed(42)
        num_confetti = 30
        colors = [self.gold, (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 255)]

        for i in range(num_confetti):
            base_x = (i * width) // num_confetti
            y_offset = (frame_count * 3 + i * 50) % (height + 100)
            x = base_x + int(20 * np.sin(frame_count * 0.1 + i))
            y = y_offset - 100

            if 0 < y < height:
                color = colors[i % len(colors)]
                angle = (frame_count + i * 10) % 360
                rect_points = cv2.boxPoints(((x, y), (8, 15), angle))
                cv2.fillPoly(image, [np.int32(rect_points)], color)

    def draw_decorative_frame(self, image):
        # draws the borders of the screen.
        height, width = image.shape[:2]
        corner_size = 60
        thickness = 8
        cv2.line(image, (10, 10), (corner_size, 10), self.gold, thickness)
        cv2.line(image, (10, 10), (10, corner_size), self.gold, thickness)
        cv2.line(
            image, (width - 10, 10), (width - corner_size, 10), self.gold, thickness
        )
        cv2.line(
            image, (width - 10, 10), (width - 10, corner_size), self.gold, thickness
        )
        cv2.line(
            image, (10, height - 10), (corner_size, height - 10), self.gold, thickness
        )
        cv2.line(
            image, (10, height - 10), (10, height - corner_size), self.gold, thickness
        )
        cv2.line(
            image,
            (width - 10, height - 10),
            (width - corner_size, height - 10),
            self.gold,
            thickness,
        )
        cv2.line(
            image,
            (width - 10, height - 10),
            (width - 10, height - corner_size),
            self.gold,
            thickness,
        )

    def draw_fancy_countdown(self, image, remaining_seconds):
        # Draws the darkened circle overlay and the large countdown numbers right before the picture is taken.
        # offloads counting functionality to the processing func
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2

        overlay = image.copy()
        cv2.circle(overlay, (center_x, center_y), 130, (0, 0, 0), -1)
        cv2.circle(overlay, (center_x, center_y), 120, self.gold, 4)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        text = str(remaining_seconds)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 6
        thickness = 10

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2

        cv2.putText(
            image,
            text,
            (text_x + 5, text_y + 5),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )
        cv2.putText(
            image, text, (text_x, text_y), font, font_scale, self.white, thickness
        )

    def process_frame(self, frame, frame_count, is_final_capture=False):
        # This ties everything (most of the functions above) together per frame

        if self.active_props.get("virtual_bg", False):
            if is_final_capture:
                frame = self.apply_high_quality_background(frame)
            else:
                frame = self.apply_virtual_background(frame)
        # Swaps the background (Live or HD depending on is_final_capture).

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if self.active_props["banner"]:
            self.draw_banner(frame)
        if self.active_props["confetti"]:
            self.draw_confetti(frame, frame_count)
        if self.active_props["frame"]:
            self.draw_decorative_frame(frame)

        face_centers = []
        # Detects the face and runs the algorithm.
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ih, iw, _ = frame.shape

                def get_pt(idx):
                    return (
                        int(face_landmarks.landmark[idx].x * iw),
                        int(face_landmarks.landmark[idx].y * ih),
                    )

                # Grab specific 3D anchors for virtual props
                # Anchors the hats to the forehead, glasses to the nose bridge, and beard/mask to the lower lip/nose.
                left_eye = get_pt(159)
                right_eye = get_pt(386)
                forehead = get_pt(10)
                nose_bridge = get_pt(168)
                lower_lip = get_pt(17)
                chin = get_pt(152)
                lower_nose = get_pt(164)
                left_cheek = get_pt(234)
                right_cheek = get_pt(454)

                # physics, basic division to check facial proportions
                face_width = np.sqrt(
                    (right_cheek[0] - left_cheek[0]) ** 2
                    + (right_cheek[1] - left_cheek[1]) ** 2
                )

                # calculate tilt
                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                angle = np.degrees(np.arctan2(dy, dx))

                # Pitch Ratio (Up/Down Detection)
                # Compares the height of the forehead to the height of the jaw
                top_dist = max(1, nose_bridge[1] - forehead[1])
                bottom_dist = max(1, chin[1] - nose_bridge[1])
                pitch_ratio = top_dist / bottom_dist

                face_centers.append(nose_bridge)

                # 1. Hats
                hat_type = self.active_props["hat"]
                if hat_type and hat_type in self.assets:
                    # Dynamically shrink the gap when looking up, expand when looking down!
                    dynamic_offset = int(face_width * 0.7 * pitch_ratio)

                    hat_x = forehead[0] + int(
                        dynamic_offset * np.sin(np.radians(angle))
                    )
                    hat_y = forehead[1] - int(
                        dynamic_offset * np.cos(np.radians(angle))
                    )

                    frame = self._overlay_image_alpha(
                        frame,
                        self.assets[hat_type],
                        hat_x,
                        hat_y,
                        angle,
                        face_width * 2.0,
                    )

                # 2. Glasses
                eyes_type = self.active_props["eyes"]
                if eyes_type and eyes_type in self.assets:
                    frame = self._overlay_image_alpha(
                        frame,
                        self.assets[eyes_type],
                        nose_bridge[0],
                        nose_bridge[1],
                        angle,
                        face_width * 1.2,
                    )

                # 3. Face Props
                face_type = self.active_props["face"]
                if face_type == "beard" and "beard" in self.assets:
                    beard_offset = int(
                        face_width * 0.08 * (0.6 / max(0.1, pitch_ratio))
                    )
                    beard_x = lower_lip[0] - int(
                        beard_offset * np.sin(np.radians(angle))
                    )
                    beard_y = lower_lip[1] + int(
                        beard_offset * np.cos(np.radians(angle))
                    )

                    frame = self._overlay_image_alpha(
                        frame,
                        self.assets["beard"],
                        beard_x,
                        beard_y,
                        angle,
                        face_width * 1.3,
                    )
                elif face_type == "mask" and "mask" in self.assets:
                    mask_offset = int(face_width * 0.05 * (0.6 / max(0.1, pitch_ratio)))
                    mask_y = lower_nose[1] + int(
                        mask_offset * np.cos(np.radians(angle))
                    )

                    frame = self._overlay_image_alpha(
                        frame,
                        self.assets["mask"],
                        lower_nose[0],
                        mask_y,
                        angle,
                        face_width * 1.2,
                    )

        # wireframe everything
        if self.active_props.get("wireframe"):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            edges = cv2.Canny(gray, 30, 100)

            wireframe_frame = np.zeros_like(frame)

            # Blue and Green channels for Cyan!
            wireframe_frame[:, :, 0] = edges
            wireframe_frame[:, :, 1] = edges

            frame = wireframe_frame
            # frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) <- if I wanted gray

        num_people = len(face_centers)
        # layers all the active UI elements and AR props on top of the user.
        if self.active_props["banner"]:
            if num_people == 1:
                title = "Future Innovator"
            elif num_people == 2:
                title = "Dynamic Duo"
            elif num_people >= 3:
                title = "SST Dream Team"
            else:
                title = "Waiting for Graduates..."

            display_text = f"--- {title} ---"
            font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            text_size = cv2.getTextSize(display_text, font, font_scale, thickness)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = 115

            cv2.putText(
                frame,
                display_text,
                (text_x + 2, text_y + 2),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
            )
            cv2.putText(
                frame,
                display_text,
                (text_x, text_y),
                font,
                font_scale,
                self.gold,
                thickness,
            )

        if self.active_props["squad_sparks"] and num_people > 1:
            self.draw_squad_connections(frame, face_centers)

        # Computes the "Halo Streaks" afterimage effect.
        if self.active_props.get("halo_streaks"):
            if (
                self.accumulated_frame is None
                or self.accumulated_frame.shape != frame.shape
            ):
                self.accumulated_frame = frame.copy().astype(np.float32)
            cv2.accumulateWeighted(
                frame.astype(np.float32), self.accumulated_frame, 0.2
            )
            frame = cv2.convertScaleAbs(self.accumulated_frame)
        else:
            self.accumulated_frame = None

        return frame

    def cycle_prop(self, category, options):
        # A helper to continuously loop through arrays of props (e.g., Cap -> Propeller -> None) when a key is pressed.
        current = self.active_props[category]
        try:
            next_idx = (options.index(current) + 1) % len(options)
        except ValueError:
            next_idx = 0
        self.active_props[category] = options[next_idx]

    def draw_controls(self, frame):
        # Renders the text menu on the left side of the screen explaining the keyboard shortcuts.
        if self.is_counting_down:
            return

        instructions = [
            "CONTROLS:",
            "1 - Cycle Hats",
            "2 - Cycle Glasses",
            "3 - Cycle Face",
            "4 - Toggle Banner",
            "5 - Toggle Confetti",
            "6 - Toggle Frame",
            "7 - Virtual Background",
            "0 - Cycle Background",
            "8 - Squad Sparks",
            "9 - Toggle Halo Streaks",
            "W - Toggle Wireframe",
            "SPACE - Capture Photo",
            "Q - Quit",
        ]

        # Semi-Transparent Rounded HUD Panel
        overlay = frame.copy()
        panel_x = 5
        panel_y = 100
        panel_w = 265
        panel_h = len(instructions) * 25 + 20

        top_left = (panel_x, panel_y)
        bottom_right = (panel_x + panel_w, panel_y + panel_h)

        # Draw a black rounded rectangle on the overlay with 15px corner radius
        self.draw_rounded_rect(
            overlay, top_left, bottom_right, (0, 0, 0), radius=15, filled=True
        )

        # Blend the overlay with the original frame with 50% opacity
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        y_offset = 130
        for i, instruction in enumerate(instructions):
            color = self.gold if i == 0 else self.white
            font = cv2.FONT_HERSHEY_SIMPLEX if i == 0 else cv2.FONT_HERSHEY_PLAIN
            size = 0.7 if i == 0 else 1.2
            thickness = 2 if i == 0 else 1

            x_pos = 15
            y_pos = y_offset + i * 25

            cv2.putText(
                frame,
                instruction,
                (x_pos + 1, y_pos + 1),
                font,
                size,
                (0, 0, 0),
                thickness + 1,
            )

            cv2.putText(
                frame, instruction, (x_pos, y_pos), font, size, color, thickness
            )

            # yeah, you probably noticed that it was drawn two times
            # its a workaround due to OpenCV not having a built-in "text outline" feature

    def print_color_ascii(self, frame, width=80):
        # Shrinks the final photo, analyzes pixel brightness and RGB values,
        # and outputs a 24-bit color ASCII art version directly into the terminal window.
        # gimicky but fun; ascii is fun yayy
        print("\n" + "=" * 60)
        print("INCOMING ASCII TRANSMISSION...")
        print("=" * 60 + "\n")
        h, w, _ = frame.shape
        aspect_ratio = h / w
        new_width = width
        new_height = int(aspect_ratio * new_width * 0.5)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        ascii_chars = "@%#*+=-:. "
        for y in range(new_height):
            line_chars = []
            for x in range(new_width):
                b, g, r = resized_frame[y, x]
                brightness = int(0.299 * r + 0.587 * g + 0.114 * b)
                char_idx = int((brightness / 255) * (len(ascii_chars) - 1))
                char = ascii_chars[char_idx]
                line_chars.append(f"\033[38;2;{r};{g};{b}m{char}")
            print("".join(line_chars) + "\033[0m")
        print("\n" + "=" * 60 + "\n")

    def save_photo(self, frame):
        # Generates a timestamp, saves the final .jpg to the graduation_photos folder, and triggers the ASCII printout.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/SST_GradTea_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✓ Photo saved: {filename}")
        self.print_color_ascii(frame, width=80)
        return filename

    def run(self):
        # The main application loop.
        # It opens the webcam, reads the frames, listens for keyboard presses (1-9, Space, Q),
        # handles the countdown timer state, and flashes the screen white when a photo is taken.

        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("=" * 60)
        print("SST GRADUATION TEA VIRTUAL PHOTOBOOTH")
        print("=" * 60)
        print("\nPhotobooth is running!")

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                processed_frame = self.process_frame(
                    frame, frame_count, is_final_capture=False
                )

                if self.is_counting_down:
                    elapsed_time = time.time() - self.countdown_start_time
                    remaining_seconds = 3 - int(elapsed_time)

                    if remaining_seconds > 0:
                        self.draw_fancy_countdown(processed_frame, remaining_seconds)
                    else:
                        print("\nProcessing high-quality photo... Please wait...")
                        final_hd_photo = self.process_frame(
                            frame.copy(), frame_count, is_final_capture=True
                        )
                        self.play_shutter_sound()
                        self.save_photo(final_hd_photo)
                        flash = np.ones_like(processed_frame) * 255
                        cv2.imshow("SST Graduation Tea Photobooth", flash)
                        cv2.waitKey(150)
                        self.is_counting_down = False

                self.draw_controls(processed_frame)
                cv2.imshow("SST Graduation Tea Photobooth", processed_frame)

                key = cv2.waitKey(1) & 0xFF

                if key in [ord("q"), ord("Q")]:
                    break
                elif key == ord("1"):
                    self.cycle_prop("hat", ["cap", "propeller", None])
                elif key == ord("2"):
                    self.cycle_prop(
                        "eyes", ["sst_glasses", "sunglasses", "heartglasses", None]
                    )
                elif key == ord("3"):
                    self.cycle_prop("face", ["beard", "mask", None])
                elif key == ord("4"):
                    self.active_props["banner"] = not self.active_props["banner"]
                elif key == ord("5"):
                    self.active_props["confetti"] = not self.active_props["confetti"]
                elif key == ord("6"):
                    self.active_props["frame"] = not self.active_props["frame"]
                elif key == ord("7"):
                    self.active_props["virtual_bg"] = not self.active_props[
                        "virtual_bg"
                    ]
                elif key == ord("0"):
                    if self.bg_images:
                        self.current_bg_idx = (self.current_bg_idx + 1) % len(
                            self.bg_images
                        )
                elif key == ord("8"):
                    self.active_props["squad_sparks"] = not self.active_props[
                        "squad_sparks"
                    ]
                elif key == ord("9"):
                    self.active_props["halo_streaks"] = not self.active_props[
                        "halo_streaks"
                    ]
                elif key in [ord("w"), ord("W")]:
                    self.active_props["wireframe"] = not self.active_props["wireframe"]
                elif key == ord(" "):
                    if not self.is_counting_down:
                        print("\nSay Cheese! Counting down...")
                        self.is_counting_down = True
                        self.countdown_start_time = time.time()

                frame_count += 1

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()
            self.segmentation.close()
            print("\nPhotobooth closed. Thank you for celebrating with us!")


def main():
    loader = Spinner("Warming up the camera and models...") # start loading animation
    loader.start() 

    try:
        photobooth = VirtualPhotobooth() # initialize photobooth class 
        loader.stop()
        print("📸 Photobooth is ready! Press 'Q' to quit.")
        photobooth.run()  

    except Exception as e:
        # stop the spinner to see error if something crashes
        loader.stop()
        print(f"\n❌ Error starting photobooth: {e}")

if __name__ == "__main__":
    main()
