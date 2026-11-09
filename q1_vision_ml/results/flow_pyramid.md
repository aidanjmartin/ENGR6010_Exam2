# Pyramidal Lucas-Kanade on real Go2 X frames

Share of Shi-Tomasi corners tracked within 1 px of the true position, mean ± standard error over 20 trials per shift (real frames, known shift in a random direction, Gaussian noise sd 2 gray levels), 21x21 px window.

| shift (px) | 0 levels | 1 levels | 2 levels | 3 levels | 4 levels |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.998 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 4 | 0.943 ± 0.002 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 8 | 0.696 ± 0.014 | 0.990 ± 0.002 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 16 | 0.275 ± 0.013 | 0.789 ± 0.008 | 0.988 ± 0.002 | 0.995 ± 0.001 | 0.995 ± 0.001 |
| 24 | 0.096 ± 0.005 | 0.524 ± 0.006 | 0.922 ± 0.005 | 0.994 ± 0.001 | 0.994 ± 0.001 |
| 32 | 0.050 ± 0.005 | 0.355 ± 0.012 | 0.767 ± 0.012 | 0.992 ± 0.001 | 0.993 ± 0.001 |
| 48 | 0.016 ± 0.002 | 0.144 ± 0.011 | 0.490 ± 0.008 | 0.947 ± 0.006 | 0.992 ± 0.002 |
| 64 | 0.007 ± 0.001 | 0.069 ± 0.006 | 0.275 ± 0.011 | 0.775 ± 0.013 | 0.992 ± 0.001 |

Consecutive real frames of the standing dog (3 pyramid levels):

| frames | corners tracked | median flow (px) | 95th percentile (px) |
| :--- | ---: | ---: | ---: |
| 1->2 | 300 | 0.07 | 0.40 |
| 2->3 | 300 | 0.07 | 0.14 |
| 3->4 | 300 | 0.09 | 0.17 |
