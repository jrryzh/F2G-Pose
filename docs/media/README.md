# Project page media

These media files were generated from the completed v6 promotional delivery at `outputs/promo/f2g_pose_v6/`. The v6 master SHA-256 is `a332ad3e2894564fb5c23318eec6cc1985ede2a474c5c6d259a833d1aaa37d31`. All MP4 files are silent H.264/yuv420p at 30 playback frames per second. Poster images were extracted from their corresponding web encodes.

| Page file | v6 source | Frames | SHA-256 |
| --- | --- | ---: | --- |
| `film.mp4` | `F2G-Pose_90s_silent.mp4`, 0–90 s | 2700 | `8632c905a83bfc30705720eef8fcf0a8a32faea8a9d064a88c0485e2da70a158` |
| `hero.mp4` | `F2G-Pose_90s_silent.mp4`, 0–6 s | 180 | `6124e682e1d57543dfe1c80c755e8d634da0787d5c683e50fa348610bd643bf2` |
| `comparison.mp4` | `F2G-Pose_90s_silent.mp4`, 31–37 s | 180 | `1824f1e3e6facbbf41413a775d26eb72a84d2d5e373271984e17396f39fc5cbd` |
| `rope_000354.mp4` | `F2G-Pose_90s_silent.mp4`, 13–19 s | 180 | `45106d336c5b2e1ab9b7ead1e9d41d3aca27f9dd508a0465de3f0a7febb09561` |
| `rope_000064.mp4` | `F2G-Pose_90s_silent.mp4`, 25–31 s | 180 | `ad214a4f14535b01b4ddff1ccdd98679432131e9d4b899d8238e5db3e59ee15b` |
| `shape_1.mp4` | `F2G-Pose_90s_silent.mp4`, 51–56 s | 150 | `b00d55c715f9dd83b1cf3bf103e96002a51a600688cd185835453dee807521cb` |
| `shape_2.mp4` | `F2G-Pose_90s_silent.mp4`, 56–61 s | 150 | `22ffe14d0af9f7a42d6e96a7fc537595e4f95d4250d7da6066f303a8fa6cd546` |
| `shape_3.mp4` | `F2G-Pose_90s_silent.mp4`, 61–66 s | 150 | `38188ab2472bc42899e1175942f1fab187fa9a4712c67db36bfcf6174abff3e3` |
| `materials.mp4` | `F2G-Pose_90s_silent.mp4`, 66–74 s | 240 | `93418b4a468396dd367689c7b3c76e920c4ed6c0f4dc92c6d93b9e749466503b` |
| `robot_keyframes.mp4` | `F2G-Pose_90s_silent.mp4`, 74–80 s | 180 | `61234f12a985d2b0482b763598ace714d8dcf8b75a5da7e418284cf82cb81736` |

The three shape excerpts show direct, saved 2,048-point SOPE network predictions selected by the user after reviewing 3,000 rotation videos: rank 2 eraser, rank 32 Chinese chess piece, and rank 43 flute. Rank 32 is also the pipeline example. No observed or GT points are fused into the prediction. Rank 2 passes all five original geometric screening thresholds. Ranks 32 and 43 are visual selections that fail some thresholds: rank 32 has 7.2% prediction outliers and 70.8% unseen-region coverage; rank 43 has 18.8% GT points beyond 5% distance and 73.9% minimum silhouette coverage. These excerpts are qualitative examples, not a strict-pass set. Full measurements and source records are in the v6 delivery.

The ROPE prediction boxes in the opening, pose chapters, comparison, grids, and recap use display-only causal three-frame temporal smoothing. Translation, dimensions, and rotation use weights [1,2,3] from oldest to current within each continuous edited interval, renormalized at the start; no future pose is read. Per-object checks fall back to the original box when projected corner-step q95 improves by less than 20%, projected box-center displacement p95 exceeds 10 pixels at 1080p, maximum center displacement exceeds 20 pixels, or visual review finds trails or misalignment. Maximum corner shift is recorded separately for visual review. Raw predictions remain unchanged. `smoothing_report.json` and the v6 pose archives in the delivery record every decision. Ground truth, point clouds, the single-frame pipeline prediction, robot keyframes, and material glass/dish shots retain their original data.

The materials excerpt contains the retained transparent red wine glass and ROPE `real-dish_003`, officially labeled specular, in source frames `000196/000060_`–`000196/000179_`. `material_v5_audit.json` in the v6 delivery checks all 120 dish frames.

The film is selected qualitative evidence. 149 categories refers to the complete ROPE evaluation. The playback rate is not model inference speed. Robot scenes comprise held recorded keyframes, not continuous control footage.
