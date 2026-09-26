# Project page media

The page uses the completed v4 edit in the private project output folder
`outputs/promo/f2g_pose_v4/`. All MP4 files here are H.264, yuv420p, silent,
and resized for web delivery. The full film is 90 seconds; the other clips are
selected excerpts or independent v4 clean clips. The page poster images were
extracted from the corresponding web encodes.

| Page file | v4 source | Content |
| --- | --- | --- |
| `film.mp4` | `F2G-Pose_90s_silent.mp4` | Full film, 0–90 s |
| `hero.mp4` | `clips/rope_000152.mp4` | ROPE scene 000152, opening visual |
| `comparison.mp4` | `clips/gt_comparison.mp4` | ROPE scene 000300, GT and prediction |
| `rope_000354.mp4` | `clips/rope_000354.mp4` | ROPE scene 000354 |
| `rope_000064.mp4` | `clips/rope_000064.mp4` | ROPE scene 000064 |
| `shape_hammer.mp4` | `clips/shape_1.mp4` | SOPE hammer, 2,048 predicted points |
| `shape_shoe.mp4` | `clips/shape_2.mp4` | SOPE shoe, 2,048 predicted points |
| `shape_helmet.mp4` | `clips/shape_3.mp4` | ROPE helmet, 2,048 predicted points |
| `materials.mp4` | master, 66–74 s | Transparent glass and specular rice cooker |
| `robot_keyframes.mp4` | master, 74–80 s | Three recorded robot cases, four held keyframes each |

For the v4 shape change, `selected_shapes.json`, `shape_selection.json`, and
`asset_manifest.csv` identify the third shape as a ROPE helmet. ROPE has no
aligned complete-shape ground truth. The material glass example is unchanged.

The 30 FPS of these MP4s is playback rate, not model inference speed. The
recorded robot keyframes are held images, not continuous control footage.
