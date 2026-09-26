# Project page media

These media files were generated from the completed v5 promotional delivery at `outputs/promo/f2g_pose_v5/`. The v5 master SHA-256 is `568697643bbd4d7e95322005881b2f727111acecb733a44c679dc190e97ba2dd`. All MP4 files are silent H.264/yuv420p at 30 playback frames per second. Poster images were extracted from their corresponding web encodes.

| Page file | v5 source | Frames | SHA-256 |
| --- | --- | ---: | --- |
| `film.mp4` | `F2G-Pose_90s_silent.mp4`, 0–90 s | 2700 | `7fae7083cc67e7325c04a3ae5cd82a14ef24fa78758acc5dda45b04c9220dd39` |
| `hero.mp4` | `F2G-Pose_90s_silent.mp4`, 0–6 s | 180 | `8f454b6c32159a2041263eedc82f2af9f483e41455524557cab0e336de45c985` |
| `comparison.mp4` | `F2G-Pose_90s_silent.mp4`, 31–37 s | 180 | `f3e53757cf921d853acaa461cbcc8d00c55fe0d0a688f50fea271a6ac9e8c3d4` |
| `rope_000354.mp4` | `F2G-Pose_90s_silent.mp4`, 13–19 s | 180 | `048b3ca44ffbabdf14c948abe638914f39be66280e415eed455178d9b17dbdd0` |
| `rope_000064.mp4` | `clips/rope_000064.mp4`, complete clean clip | 210 | `153d227b12154d5c74434f10739c3ba0cf43cc32ff8611bf2d67372796a01548` |
| `shape_1.mp4` | `F2G-Pose_90s_silent.mp4`, 51–56 s | 150 | `b00d55c715f9dd83b1cf3bf103e96002a51a600688cd185835453dee807521cb` |
| `shape_2.mp4` | `F2G-Pose_90s_silent.mp4`, 56–61 s | 150 | `22ffe14d0af9f7a42d6e96a7fc537595e4f95d4250d7da6066f303a8fa6cd546` |
| `shape_3.mp4` | `F2G-Pose_90s_silent.mp4`, 61–66 s | 150 | `38188ab2472bc42899e1175942f1fab187fa9a4712c67db36bfcf6174abff3e3` |
| `materials.mp4` | `F2G-Pose_90s_silent.mp4`, 66–74 s | 240 | `93418b4a468396dd367689c7b3c76e920c4ed6c0f4dc92c6d93b9e749466503b` |
| `robot_keyframes.mp4` | `F2G-Pose_90s_silent.mp4`, 74–80 s | 180 | `61234f12a985d2b0482b763598ace714d8dcf8b75a5da7e418284cf82cb81736` |

The three shape excerpts show direct, saved 2,048-point SOPE network predictions selected by the user after reviewing 3,000 rotation videos: rank 2 eraser, rank 32 Chinese chess piece, and rank 43 flute. Rank 32 is also the pipeline example. No observed or GT points are fused into the prediction. Rank 2 passes all five original geometric screening thresholds. Ranks 32 and 43 are visual selections that fail some thresholds: rank 32 has 7.2% prediction outliers and 70.8% unseen-region coverage; rank 43 has 18.8% GT points beyond 5% distance and 73.9% minimum silhouette coverage. These excerpts are qualitative examples, not a strict-pass set. Full measurements and source records are in the private v5 delivery.

The materials excerpt contains the retained transparent red wine glass and ROPE `real-dish_003`, officially labeled specular, in source frames `000196/000060_`–`000196/000179_`. `material_v5_audit.json` in the private delivery checks all 120 dish frames.

The film is selected qualitative evidence. 149 categories refers to the complete ROPE evaluation. The playback rate is not model inference speed. Robot scenes comprise held recorded keyframes, not continuous control footage.
