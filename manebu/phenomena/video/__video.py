import numpy as np
import jax
import jax.numpy as jnp
import cv2.cv2 as cv2
from . import utils

from pathlib import Path


class VideoObject:
    def __init__(self, video_path: Path, reduction_k=1):
        self.sparse_mat = None
        self.low_rank_mat = None
        self.path = video_path

        cap = cv2.VideoCapture(str(self.path))
        self.w, self.h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.w, self.h = int(self.w / reduction_k), int(self.h / reduction_k)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.frames = []
        for i in range(self.frame_count):
            is_succeed, frame = cap.read()
            if not is_succeed:
                break
            self.frames.append(
                cv2.resize(
                    frame, (self.w, self.h)
                )
            )

        cap.release()

        self.frames = np.array(self.frames).T

    def RPCA(self, max_iter=500):
        L, S = utils.video_robust_pca(
            jnp.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).flatten() for f in self.frames.T]).T, max_iter=max_iter
        )
        if isinstance(L, jax.Array) and isinstance(S, jax.Array):
            self.low_rank_mat, self.sparse_mat = L.block_until_ready(), S.block_until_ready()
        else:
            self.low_rank_mat, self.sparse_mat = np.array(L), np.array(S)

        def _video_write(frames, output_path, fps, width, height):
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), codec, fps, (int(width), int(height)))
            for i in range(frames.shape[1]):
                buf = frames[:, i]
                lmin, lmax = np.abs(buf.min()), np.abs(buf.max())
                buf = np.clip(((buf + lmin) / (lmin + lmax) * 255), 0, 255)
                buf = cv2.cvtColor(buf.reshape(int(height), -1).astype("uint8"), cv2.COLOR_GRAY2BGR)
                writer.write(buf)
            writer.release()

        _video_write(np.array(self.low_rank_mat), f"./rpca/{self.path.stem}_LowRank.mp4", self.fps, self.w, self.h)
        _video_write(np.array(self.sparse_mat), f"./rpca/{self.path.stem}_Sparse.mp4", self.fps, self.w, self.h)
        return self.low_rank_mat, self.sparse_mat

    def moving_detection(self, output_path: Path):
        if self.low_rank_mat is None:
            self.low_rank_mat, self.sparse_mat = self.RPCA()

        codec = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), codec, self.fps, (self.w, self.h))
        frames = np.array(self.low_rank_mat)
        kernel_size = min(self.w // 5 + 1, self.h // 5 + 1)
        kernel_size = (kernel_size, kernel_size)
        color_frames = self.frames
        for i in range(frames.shape[1]):
            mask = frames[:, i]
            mask += np.abs(mask.min())
            _tmp = mask.copy()
            threshold = np.mean(_tmp)
            mask[_tmp < threshold] = 255
            mask[_tmp >= threshold] = 0
            mask = cv2.GaussianBlur(mask, kernel_size, 0)
            mask = cv2.cvtColor(
                mask.reshape((self.h, -1)),
                cv2.COLOR_GRAY2RGB
            )
            color = color_frames[:, :, :, i].T
            buf =  (color * (mask / 255))
            buf /= buf.max()
            buf *= 255
            writer.write(buf.astype("uint8"))
        writer.release()
