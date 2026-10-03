# coding: utf-8
"""Morphology-based post-processing for segmentation masks.

What this module roughly does:
  1) Remove tiny connected components per class (reduce false positives)
  2) Optional skeletonization for guidewire (helps thin structures)
  3) Optional dilation after skeletonization (bring the width back)
"""

import cv2
import numpy as np
from skimage.morphology import remove_small_objects, binary_dilation, skeletonize


class MorphologyPostprocessor:
    """Simple post-processor for masks with labels {0, 1, 2}."""

    def __init__(
        self,
        min_area_catheter: int = 50,
        min_area_guidewire: int = 30,
        guidewire_skeleton: bool = False,
        guidewire_dilate_width: int = 2,
    ):
        """Create a post-processor.

        Args:
            min_area_catheter: Minimum connected-component area for catheter (class 1)
            min_area_guidewire: Minimum connected-component area for guidewire (class 2)
            guidewire_skeleton: If True, skeletonize guidewire masks (thin to ~1px)
            guidewire_dilate_width: Width (in pixels) to dilate back after skeletonization
        """
        self.min_area_catheter = min_area_catheter
        self.min_area_guidewire = min_area_guidewire
        self.guidewire_skeleton = guidewire_skeleton
        self.guidewire_dilate_width = guidewire_dilate_width

    def __call__(self, mask_pred: np.ndarray) -> np.ndarray:
        """Run post-processing on a single predicted mask.

        Args:
            mask_pred: [H, W] predicted mask with values 0/1/2
                      (0=background, 1=catheter, 2=guidewire)

        Returns:
            [H, W] post-processed mask
        """
        result = mask_pred.copy()

        # Step 1: process Catheter (class 1)
        result = self._process_class(
            result,
            class_id=1,
            min_area=self.min_area_catheter,
            skeleton=False,
        )

        # Step 2: process Guidewire (class 2)
        result = self._process_class(
            result,
            class_id=2,
            min_area=self.min_area_guidewire,
            skeleton=self.guidewire_skeleton,
            dilate_width=self.guidewire_dilate_width if self.guidewire_skeleton else None,
        )

        return result

    def _process_class(
        self,
        mask: np.ndarray,
        class_id: int,
        min_area: int,
        skeleton: bool = False,
        dilate_width: int = None,
    ) -> np.ndarray:
        """Process a single class and write it back into the label map."""

        # Extract a binary mask for this class.
        class_mask = (mask == class_id).astype(np.uint8)

        # Step 1: remove tiny connected components.
        class_mask = remove_small_objects(
            class_mask.astype(bool),
            min_size=min_area
        ).astype(np.uint8)

        # Step 2: skeletonize + dilate back (optional; mostly for guidewire).
        if skeleton and dilate_width is not None:
            class_mask = self._skeleton_process(class_mask, dilate_width)

        # Step 3: update the original label map.
        mask[mask == class_id] = 0
        mask[class_mask == 1] = class_id

        return mask

    def _skeleton_process(
        self,
        class_mask: np.ndarray,
        dilate_width: int,
    ) -> np.ndarray:
        """Skeletonize to ~1px and then dilate back to a chosen thickness."""

        # Skeletonize (thinning).
        skeleton = skeletonize(class_mask.astype(bool))

        # Dilate back to the requested width.
        iterations = max(1, dilate_width // 2)
        skeleton_dilated = binary_dilation(skeleton, iterations=iterations)

        return skeleton_dilated.astype(np.uint8)


def postprocess_batch(
    masks: np.ndarray,
    min_area_catheter: int = 50,
    min_area_guidewire: int = 30,
    guidewire_skeleton: bool = False,
    guidewire_dilate_width: int = 2,
) -> np.ndarray:
    """Post-process one mask or a batch of masks."""
    processor = MorphologyPostprocessor(
        min_area_catheter=min_area_catheter,
        min_area_guidewire=min_area_guidewire,
        guidewire_skeleton=guidewire_skeleton,
        guidewire_dilate_width=guidewire_dilate_width,
    )

    if masks.ndim == 2:
        # Single mask.
        return processor(masks)
    elif masks.ndim == 3:
        # Batch of masks.
        result = np.zeros_like(masks)
        for i in range(masks.shape[0]):
            result[i] = processor(masks[i])
        return result
    else:
        raise ValueError(f"Expected 2D or 3D array, got {masks.ndim}D")


# =============================================================================
# Examples
# =============================================================================

if __name__ == '__main__':
    # Example 1: single mask.
    print("Example 1: single mask")

    # Pretend we have a predicted mask.
    mask_pred = np.random.randint(0, 3, (512, 512))

    processor = MorphologyPostprocessor(
        min_area_catheter=50,
        min_area_guidewire=30,
        guidewire_skeleton=False,
    )

    mask_post = processor(mask_pred)
    print(f"Original mask shape: {mask_pred.shape}")
    print(f"Post-processed mask shape: {mask_post.shape}")
    print(f"Foreground pixels: {(mask_pred > 0).sum()} -> {(mask_post > 0).sum()}")

    # Example 2: enable skeletonization.
    print("\nExample 2: with skeletonization")

    processor_skeleton = MorphologyPostprocessor(
        min_area_catheter=50,
        min_area_guidewire=30,
        guidewire_skeleton=True,
        guidewire_dilate_width=2,
    )

    mask_post_skeleton = processor_skeleton(mask_pred)
    print(f"Foreground pixels after skeletonization: {(mask_post_skeleton > 0).sum()}")

    # Example 3: batch processing.
    print("\nExample 3: batch processing")

    masks_batch = np.random.randint(0, 3, (8, 512, 512))
    masks_post_batch = postprocess_batch(
        masks_batch,
        min_area_catheter=50,
        min_area_guidewire=30,
        guidewire_skeleton=False,
    )
    print(f"Processed {masks_batch.shape[0]} masks")
    print(
        f"Foreground pixels (avg): {(masks_batch > 0).sum() / masks_batch.shape[0]:.0f} "
        f"-> {(masks_post_batch > 0).sum() / masks_post_batch.shape[0]:.0f}"
    )

