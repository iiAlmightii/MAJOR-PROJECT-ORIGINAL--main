import os
import asyncio
from typing import Optional, Dict, Any


async def extract_video_metadata(file_path: str) -> Dict[str, Any]:
    """Extract video metadata using OpenCV."""
    try:
        import cv2
        loop = asyncio.get_event_loop()

        def _extract():
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return {}
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            return {
                "fps": round(fps, 2),
                "width": width,
                "height": height,
                "duration": round(duration, 2),
                "frame_count": int(frame_count),
            }

        return await loop.run_in_executor(None, _extract)
    except Exception as e:
        return {"error": str(e)}


async def generate_thumbnail(file_path: str, video_id: str) -> Optional[str]:
    """Generate a thumbnail at 10% into the video."""
    try:
        import cv2
        from app.config import settings

        thumb_dir = os.path.join(settings.UPLOAD_DIR, "thumbnails")
        os.makedirs(thumb_dir, exist_ok=True)
        thumb_path = os.path.join(thumb_dir, f"{video_id}.jpg")

        loop = asyncio.get_event_loop()

        def _generate():
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return None
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.1))
            ret, frame = cap.read()
            cap.release()
            if ret:
                # Resize to 480p width
                h, w = frame.shape[:2]
                new_w = 480
                new_h = int(h * new_w / w)
                resized = cv2.resize(frame, (new_w, new_h))
                cv2.imwrite(thumb_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return thumb_path
            return None

        return await loop.run_in_executor(None, _generate)
    except Exception:
        return None
