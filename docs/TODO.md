# TODO

Open items that need a person, not a code change. Remove an item when it is done.

## Retake the web interface screenshot (`docs/assets/ui-editor.png`)

**Why.** The README shows `docs/assets/ui-editor.png`. It predates the MCAP input (task_0033): it has no Rectification lane between Source and Detectors and no topic picker, and its alt text in `README.md` says the stages run "from source to semantic segmentation". A reader comparing it with the running interface sees a different layout.

**What to do.**

1. Start the interface from the repository root with the GPU stack, so the model cards are solid like in the current image:
   ```bash
   utils/run_web.sh --gpu
   ```
   Without `--gpu` the interface works too, but every model card shows **Not ready**.
2. Open <http://127.0.0.1:8000>, click **Open YAML** and choose `configs/rfdetr_sam2_lanes_motion_da3.yaml` (the config in the current image).
3. Click **Prepare all** if the weights are not downloaded yet (about 3.1 GB).
4. Drop a recording on the page and select a three-second segment, as in the current image. Use an `.mcap` with an image topic to show the topic picker and the Rectification lane; a video hides both.
   - The `CameraInfo` badge and the "use the CameraInfo" tick need a topic that has a `CameraInfo`.
   - The Rectification switch is enabled only for a topic whose `CameraInfo` has distortion.
5. Open the Kalman smoothing settings in the side panel, as in the current image.
6. Take the screenshot at the same size as the current file (`file docs/assets/ui-editor.png` shows its dimensions) and overwrite `docs/assets/ui-editor.png`.
7. Update the image alt text in `README.md` if the stages or panels shown changed. The alt text must describe what is in the picture.
8. Optionally add a second screenshot of the topic picker (for example `docs/assets/ui-mcap.png`) and one line under the MCAP sentence in the README requirements section.

**Check.** Run `git diff --stat` and make sure only the image files and the README alt text changed. Do not commit screenshots that show private file paths or recordings you cannot publish.
