## 2026-10-07-sonnet-5-5-eval45

- episodes 45, mean reward 0.467, median 0.325
- predicted mean 0.443; Spearman(predicted, actual) 0.08
- reward < 0.7 on 33/45; at the optimum on 4
- feasible plans cost 17% more than the optimum (median), 42% (mean)
- where the extra cost went (sum over feasible plans, plan minus optimum): delay +7800, unserved +2880, handed_on +242; cancellations +37, short-turns -80
- real tier: cheaper than the real dispatchers' outcome on 2/4 feasible plans

| task | tier | runs | predicted | reward | cost | optimum | floor |
|---|---|---:|---:|---:|---:|---:|---:|
| gnlw-2025-03-10-0700-210m-blockage-0 | blockage | 34 | 0.52 | 0.200 | 2463 | 455 | 386 |
| gnlw-2025-01-20-1800-180m-compound-0 | compound | 24 | 0.26 | 0.200 | 1735 | 664 | 539 |
| gnlw-2025-04-25-1730-210m-blockage-0 | blockage | 33 | 0.57 | 0.200 | 1890 | 1370 | 1355 |
| gnlw-2025-02-14-1730-210m-blockage-0 | blockage | 33 | 0.29 | 0.200 | 1773 | 875 | 776 |
| gnlw-2025-02-12-0900-180m-real-0 | real | 31 | 0.19 | 0.200 | 1461 | 643 | 529 |
| gnlw-2025-02-16-1730-240m-compound-0 | compound | 19 | 0.45 | 0.205 | 917 | 656 | 652 |
| gnlw-2025-03-12-1730-210m-loop-0 | loop | 33 | 0.44 | 0.212 | 393 | 166 | 158 |
| gnlw-2025-05-30-0630-180m-blockage-0 | blockage | 29 | 0.40 | 0.214 | 796 | 454 | 385 |
| gnlw-2025-02-16-1530-210m-compound-0 | compound | 18 | 0.53 | 0.216 | 475 | 273 | 269 |
| gnlw-2025-05-31-0800-210m-compound-0 | compound | 32 | 0.60 | 0.221 | 1658 | 1179 | 1017 |
| gnlw-2025-03-12-1530-180m-blockage-0 | blockage | 30 | 0.47 | 0.221 | 1125 | 806 | 730 |
| gnlw-2025-04-26-1700-180m-blockage-0 | blockage | 30 | 0.42 | 0.225 | 645 | 447 | 433 |
| gnlw-2025-02-15-0700-240m-compound-0 | compound | 32 | 0.52 | 0.225 | 1158 | 858 | 785 |
| gnlw-2025-02-13-0954-180m-real-0 | real | 30 | 0.40 | 0.251 | 334 | 172 | 154 |
| gnlw-2025-05-26-1630-150m-blockage-0 | blockage | 26 | 0.57 | 0.253 | 484 | 254 | 185 |
| gnlw-2025-02-13-1530-180m-blockage-0 | blockage | 30 | 0.43 | 0.258 | 858 | 648 | 588 |
| gnlw-2025-02-15-0630-180m-blockage-0 | blockage | 20 | 0.30 | 0.263 | 2657 | 2300 | 2119 |
| gnlw-2025-04-23-1630-150m-blockage-0 | blockage | 26 | 0.46 | 0.265 | 2232 | 1926 | 1782 |
| gnlw-2025-04-26-0630-180m-blockage-0 | blockage | 20 | 0.23 | 0.269 | 1252 | 1047 | 980 |
| gnlw-2025-05-28-1830-210m-compound-0 | compound | 31 | 0.58 | 0.280 | 612 | 436 | 383 |
| gnlw-2025-05-27-1700-180m-blockage-0 | blockage | 30 | 0.53 | 0.312 | 1008 | 791 | 670 |
| gnlw-2025-04-27-0830-210m-blockage-0 | blockage | 18 | 0.25 | 0.315 | 1380 | 1283 | 1283 |
| gnlw-2025-02-15-1630-150m-loop-0 | loop | 26 | 0.38 | 0.325 | 406 | 275 | 234 |
| gnlw-2025-03-16-1630-180m-compound-0 | compound | 15 | 0.51 | 0.331 | 1352 | 1259 | 1256 |
| gnlw-2025-01-25-0730-180m-compound-0 | compound | 26 | 0.49 | 0.333 | 1467 | 1302 | 1218 |
| gnlw-2025-01-26-1700-210m-compound-0 | compound | 16 | 0.27 | 0.352 | 726 | 618 | 588 |
| gnlw-2025-01-21-0626-180m-real-0 | real | 29 | 0.53 | 0.374 | 589 | 445 | 356 |
| gnlw-2025-01-23-0800-210m-compound-0 | compound | 34 | 0.47 | 0.395 | 1002 | 894 | 841 |
| gnlw-2025-01-23-0900-180m-compound-0 | compound | 30 | 0.42 | 0.411 | 1019 | 853 | 704 |
| gnlw-2025-03-11-0730-150m-blockage-0 | blockage | 26 | 0.60 | 0.427 | 2227 | 2043 | 1851 |
| gnlw-2025-01-25-1700-180m-blockage-0 | blockage | 30 | 0.36 | 0.674 | 358 | 320 | 275 |
| gnlw-2025-05-26-1800-150m-blockage-0 | blockage | 25 | 0.50 | 0.682 | 1051 | 1014 | 968 |
| gnlw-2025-05-27-1830-180m-blockage-0 | blockage | 27 | 0.59 | 0.692 | 629 | 596 | 560 |
| gnlw-2025-01-22-1600-240m-compound-0 | compound | 38 | 0.54 | 0.719 | 701 | 670 | 627 |
| gnlw-2025-01-22-1800-150m-loop-0 | loop | 25 | 0.58 | 0.744 | 428 | 397 | 336 |
| gnlw-2025-04-27-0800-210m-compound-0 | compound | 15 | 0.32 | 0.777 | 399 | 382 | 378 |
| gnlw-2025-05-27-0830-210m-blockage-0 | blockage | 34 | 0.50 | 0.832 | 810 | 795 | 768 |
| gnlw-2025-01-25-0630-180m-loop-0 | loop | 20 | 0.46 | 0.871 | 126 | 116 | 102 |
| gnlw-2025-02-11-0900-150m-blockage-0 | blockage | 23 | 0.41 | 0.941 | 729 | 725 | 721 |
| gnlw-2025-03-16-1700-210m-compound-0 | compound | 16 | 0.25 | 0.953 | 532 | 529 | 529 |
| gnlw-2025-05-26-0730-150m-blockage-0 | blockage | 26 | 0.46 | 0.985 | 858 | 857 | 848 |
| gnlw-2025-02-16-1700-210m-compound-0 | compound | 16 | 0.53 | 1.000 | 796 | 796 | 787 |
| gnlw-2025-03-13-0706-180m-real-0 | real | 26 | 0.50 | 1.000 | 378 | 378 | 378 |
| gnlw-2025-05-31-0900-180m-compound-0 | compound | 30 | 0.36 | 1.000 | 226 | 226 | 216 |
| gnlw-2025-06-22-1600-240m-compound-0 | compound | 18 | 0.51 | 1.000 | 668 | 668 | 668 |
