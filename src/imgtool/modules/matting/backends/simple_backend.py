"""简易抠图后端：边界背景建模 + GrabCut，零权重依赖。

质量不如深度学习模型，但装完即可用，用来保证“导入 → 抠图 → 预览 → 保存”链路始终可跑通。

流程：
1. 缩放工作分辨率；
2. 用画面四周的像素拟合平滑背景，得到「与背景的色差图」；
3. 对色差图做 Otsu 阈值分割，得到粗略前景；
4. 把粗略结果与用户的前景标记作为 GrabCut 的初始化掩码做细化。
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import MattingBackend


class SimpleBackend(MattingBackend):
    name = "simple"
    label = "简易（GrabCut，无需权重）"
    post_cleanup = True

    def __init__(self, params: dict | None = None, device: str = "auto") -> None:
        super().__init__(params, device)
        self.max_size = int(self.params.get("max_size") or 1600)
        self.iterations = int(self.params.get("grabcut_iters") or 4)
        self.margin_ratio = float(self.params.get("grabcut_margin") or 0.02)
        # GrabCut 比较慢，工作分辨率封顶
        self.work_limit = int(self.params.get("grabcut_work_limit") or 1200)

    # ------------------------------------------------------------------
    def predict(self, rgb: np.ndarray, hint: np.ndarray | None = None) -> np.ndarray:
        h, w = rgb.shape[:2]
        limit = max(64, min(int(self.max_size), int(self.work_limit)))
        scale = min(1.0, float(limit) / float(max(h, w)))
        if scale < 1.0:
            work = cv2.resize(
                rgb,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            work = rgb

        sh, sw = work.shape[:2]
        work_hint = None
        if hint is not None:
            work_hint = cv2.resize(hint, (sw, sh), interpolation=cv2.INTER_LINEAR)
            if float(work_hint.max()) <= 0:
                work_hint = None

        # 主路径：全自动（背景光滑时，靠背景外推 + 色差分割最准）
        rough, use_mask_init = self._rough_foreground(work)

        alpha_small = self._refine_with_grabcut(
            work, rough, use_mask_init, hint=work_hint
        )
        if work_hint is not None:
            # 复杂背景上 GrabCut 可能把笔迹旁整块同色背景连进主体。
            # 以自动结果作底，只允许笔迹附近新增前景；补漏需要在漏抠处补涂。
            baseline = self._refine_with_grabcut(work, rough, use_mask_init)
            seed = np.where(work_hint > 0.5, 255, 0).astype(np.uint8)
            radius = min(40, max(12, int(round(min(sh, sw) * 0.06))))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            nearby = cv2.dilate(seed, kernel).astype(np.float32) / 255.0
            alpha_small = np.maximum(baseline, alpha_small * nearby)

        if scale < 1.0:
            alpha = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            alpha = alpha_small
        return np.clip(alpha, 0.0, 1.0)

    # ------------------------------------------------------------------
    def _rough_foreground(self, rgb: np.ndarray) -> tuple[np.ndarray, bool]:
        """返回 (粗略前景掩码, 是否可用作 GrabCut 的掩码初始化)。

        背景足够平滑（多项式拟合残差小）时用色差分割；否则交给矩形初始化。
        """
        sh, sw = rgb.shape[:2]
        bg, residual = self._background_estimate(rgb)
        if residual > 12.0:  # 背景不光滑，多项式外推不可信
            return np.zeros((sh, sw), np.uint8), False

        rough = self._mask_from_background(rgb, bg)
        ratio = float(np.count_nonzero(rough)) / float(max(1, sh * sw))
        if not (0.002 < ratio < 0.98):
            return np.zeros((sh, sw), np.uint8), False
        return rough, True

    # ------------------------------------------------------------------
    def _border_mask(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        bw = max(3, int(round(min(h, w) * 0.06)))
        mask = np.zeros((h, w), np.uint8)
        mask[:bw, :] = 255
        mask[-bw:, :] = 255
        mask[:, :bw] = 255
        mask[:, -bw:] = 255
        return mask

    def _background_estimate(self, rgb: np.ndarray, degree: int = 2) -> tuple[np.ndarray, float]:
        """用四周边缘像素拟合一张平滑的背景（多项式），返回 (背景图, 边缘拟合残差)。

        适合渐变/棚拍类背景：背景在边缘可见，于是可以外推。
        残差大说明背景不光滑（例如杂乱实景），此时调用方应改用矩形初始化。
        """
        h, w = rgb.shape[:2]
        scale = min(1.0, 320.0 / float(max(h, w)))
        if scale < 1.0:
            small = cv2.resize(
                rgb,
                (max(16, int(round(w * scale))), max(16, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = rgb
        sh, sw = small.shape[:2]

        ys, xs = np.mgrid[0:sh, 0:sw]
        xn = (xs.astype(np.float64) / max(1, sw - 1)) * 2.0 - 1.0
        yn = (ys.astype(np.float64) / max(1, sh - 1)) * 2.0 - 1.0
        terms = [np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn]
        if degree >= 3:
            terms += [xn ** 3, yn ** 3, xn * xn * yn, xn * yn * yn]
        design = np.stack([t.ravel() for t in terms], axis=1)

        bw = max(2, int(round(min(sh, sw) * 0.06)))
        border = np.zeros((sh, sw), bool)
        border[:bw, :] = True
        border[-bw:, :] = True
        border[:, :bw] = True
        border[:, -bw:] = True
        sel = border.ravel()

        flat = small.reshape(-1, 3).astype(np.float64)
        a_border, b_border = design[sel], flat[sel]

        coef, *_ = np.linalg.lstsq(a_border, b_border, rcond=None)
        for _ in range(2):  # 剔除离群边缘（例如边缘上恰好有主体）
            residual = np.abs(a_border @ coef - b_border).max(axis=1)
            limit = max(4.0, float(np.percentile(residual, 90)) * 2.0)
            keep = residual <= limit
            if int(keep.sum()) < design.shape[1] * 6:
                break
            coef, *_ = np.linalg.lstsq(a_border[keep], b_border[keep], rcond=None)

        residual = np.abs(a_border @ coef - b_border).mean()
        bg_small = np.clip((design @ coef).reshape(sh, sw, 3), 0, 255).astype(np.uint8)
        if (sh, sw) != (h, w):
            bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            bg = bg_small
        return bg, float(residual)

    def _mask_from_background(self, rgb: np.ndarray, bg: np.ndarray) -> np.ndarray:
        """比较原图与外推背景的色差，返回 0/255 的粗略前景掩码。"""
        sh, sw = rgb.shape[:2]
        lab_a = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_b = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB).astype(np.float32)
        dist = np.linalg.norm(lab_a - lab_b, axis=2)

        # 鲁棒归一化 → Otsu
        p99 = float(np.percentile(dist, 99.0))
        if p99 < 1e-3:
            return np.zeros((sh, sw), np.uint8)
        norm = np.clip(dist / p99 * 255.0, 0, 255).astype(np.uint8)
        norm = cv2.GaussianBlur(norm, (0, 0), 1.5)
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        return binary

    def _background_distance_mask(self, rgb: np.ndarray) -> np.ndarray:
        """（调试用）只做色差分割，不跑 GrabCut。"""
        bg, _ = self._background_estimate(rgb)
        return self._mask_from_background(rgb, bg)

    def _run_grabcut(
        self, bgr: np.ndarray, mask: np.ndarray, rect: tuple, mode: int
    ) -> np.ndarray:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        working = mask.copy()
        cv2.grabCut(bgr, working, rect, bgd, fgd, self.iterations, mode)
        return np.where((working == cv2.GC_FGD) | (working == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    @staticmethod
    def _degenerate(mask: np.ndarray) -> bool:
        if not mask.any():
            return True
        return np.count_nonzero(mask) >= 0.995 * mask.size

    def _refine_with_grabcut(
        self,
        rgb: np.ndarray,
        rough: np.ndarray,
        use_mask_init: bool,
        hint: np.ndarray | None = None,
    ) -> np.ndarray:
        sh, sw = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ratio = float(np.count_nonzero(rough)) / float(max(1, sh * sw))
        border = self._border_mask((sh, sw)) > 0
        has_hint = hint is not None and float(hint.max()) > 0
        seed = np.where(hint > 0.5, 255, 0).astype(np.uint8) if has_hint else None
        core = cv2.erode(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) if has_hint else None
        if has_hint and not core.any():
            core = seed

        fg: np.ndarray | None = None
        try:
            if use_mask_init and 0.002 < ratio < 0.98:
                # 色差给出轮廓先验，画笔是确定前景；两者共同参与 GrabCut。
                mask = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
                mask[rough > 0] = cv2.GC_PR_FGD
                mask[border] = cv2.GC_BGD
                if has_hint:
                    mask[seed > 0] = cv2.GC_PR_FGD
                    mask[core > 0] = cv2.GC_FGD
                fg = self._run_grabcut(bgr, mask, (0, 0, 1, 1), cv2.GC_INIT_WITH_MASK)
            elif has_hint:
                # 复杂背景：笔迹只表示确定前景，周围仍允许算法扩展主体。
                # 不能把笔迹外全部设为确定背景，否则只能抠出笔刷形状。
                margin = max(8, int(round(min(sh, sw) * 0.12)))
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
                search = cv2.dilate(seed, k)
                mask = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
                mask[search > 0] = cv2.GC_PR_FGD
                mask[border] = cv2.GC_BGD
                mask[seed > 0] = cv2.GC_PR_FGD
                mask[core > 0] = cv2.GC_FGD
                fg = self._run_grabcut(bgr, mask, (0, 0, 1, 1), cv2.GC_INIT_WITH_MASK)
            else:
                # 背景杂乱：给一个居中的椭圆先验。
                # 比"整块矩形"稳得多——矩形把画面四周的杂色也当成可能前景，容易被带偏。
                mask = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
                cv2.ellipse(
                    mask,
                    (sw // 2, sh // 2),
                    (max(4, int(sw * 0.30)), max(4, int(sh * 0.36))),
                    0,
                    0,
                    360,
                    cv2.GC_PR_FGD,
                    -1,
                )
                mask[border] = cv2.GC_BGD
                fg = self._run_grabcut(bgr, mask, (0, 0, 1, 1), cv2.GC_INIT_WITH_MASK)

            if (fg is None or self._degenerate(fg)) and not has_hint:
                # 兜底：退回最经典的矩形初始化
                margin = max(2, int(round(min(sh, sw) * self.margin_ratio)))
                rect = (margin, margin, max(1, sw - 2 * margin), max(1, sh - 2 * margin))
                fg = self._run_grabcut(bgr, np.zeros((sh, sw), np.uint8), rect, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            fg = None

        if fg is None or self._degenerate(fg):
            # 有笔迹时不能退回与未涂抹相同的结果，更不能整图保留。
            if has_hint:
                fg = np.where(hint > 0.5, 255, 0).astype(np.uint8)
            else:
                fg = rough if rough.any() else np.full((sh, sw), 255, np.uint8)

        return fg.astype(np.float32) / 255.0
