# Four-Method Confirmation Comparison

All 12 tasks retained. Native: 2 training seeds x 3 repeats per task/condition; SONIC: 3 released-policy repeats, no training-seed dimension. No native/SONIC pooled mean. Completion is full-motion completion under the common protocol. Arm qualification and SONIC per-rollout strict qualification are deliberately not compared here.

All errors are cm. Time-p95 is the mean of per-rollout time-p95, not a pooled quantile; last-20% tail is a different metric, floor(0.8*N)..N-1. Failed full-motion rows make full-horizon and tail aggregates undefined, even if a controller kept recording. No complete-case filtering. Prefix is separately checked at source-clock frames 0..min(N-1, floor(fps*prefix_seconds)), including phase zero, with identical bodies and definitions.

## Nominal

| Metric | BeyondMimic | Static Search | MimicX | Released SONIC |
| --- | --- | --- | --- | --- |
| Full-motion completion | 60/72 | 66/72 | 66/72 | 30/36 |
| Full horizon World mean cm | undefined | undefined | undefined | undefined |
| Full horizon World time-p95 cm | undefined | undefined | undefined | undefined |
| Full horizon Local mean cm | undefined | undefined | undefined | undefined |
| Full horizon Local time-p95 cm | undefined | undefined | undefined | undefined |
| Common prefix ONLY World mean cm | undefined | 3.4693 | 3.6949 | 5.2552 |
| Common prefix ONLY World time-p95 cm | undefined | 5.3279 | 5.4426 | 8.5669 |
| Common prefix ONLY Local mean cm | undefined | 2.5475 | 2.5901 | 3.2172 |
| Common prefix ONLY Local time-p95 cm | undefined | 3.4188 | 3.4999 | 5.1109 |
| Last 20% world mean cm | undefined (censored) | undefined (censored) | undefined (censored) | undefined (censored) |
| Last 20% local mean cm | undefined (censored) | undefined (censored) | undefined (censored) | undefined (censored) |

## Perturbed

| Metric | BeyondMimic | Static Search | MimicX | Released SONIC |
| --- | --- | --- | --- | --- |
| Full-motion completion | 60/72 | 66/72 | 66/72 | 30/36 |
| Full horizon World mean cm | undefined | undefined | undefined | undefined |
| Full horizon World time-p95 cm | undefined | undefined | undefined | undefined |
| Full horizon Local mean cm | undefined | undefined | undefined | undefined |
| Full horizon Local time-p95 cm | undefined | undefined | undefined | undefined |
| Common prefix ONLY World mean cm | undefined | 3.3968 | 3.6024 | 5.3180 |
| Common prefix ONLY World time-p95 cm | undefined | 5.1661 | 5.3115 | 8.6830 |
| Common prefix ONLY Local mean cm | undefined | 2.5644 | 2.6304 | 3.1960 |
| Common prefix ONLY Local time-p95 cm | undefined | 3.4376 | 3.5305 | 5.0876 |
| Last 20% world mean cm | undefined (censored) | undefined (censored) | undefined (censored) | undefined (censored) |
| Last 20% local mean cm | undefined (censored) | undefined (censored) | undefined (censored) | undefined (censored) |

## Training and Assigned-Wall Costs

| Metric | BeyondMimic | Static Search | MimicX | Released SONIC |
| --- | --- | --- | --- | --- |
| PPO updates per endpoint | 3000 continuation + 3000 shared donor | 3000 continuation + 3000 shared donor | 3000 continuation + 3000 shared donor | 0 |
| New continuation PPO updates | 72000 | 72000 | 72000 | 0 |
| Shared donor pool updates (reused, counted once) | 72000 | 72000 | 72000 | not applicable |
| Assigned-GPU phase wall hours | 42.5017 | 42.9105 | 42.7109 | 1.1358 |
| Phase scope | continuation training + selection + evaluation | continuation training + selection + evaluation | continuation training + selection + evaluation | released-policy evaluation |
| Measured SONIC physical assignments | not applicable | not applicable | not applicable | 72/72 |
| Observed SONIC evaluation wall h (covered subset ONLY) | see phase above | see phase above | see phase above | 1.1358 |
| Observed SONIC prelaunch assigned overhead h (separate) | not applicable | not applicable | not applicable | 0.0064 |
| Upstream pretraining cost | not estimated here | not estimated here | not estimated here | unknown, not zero |
| GPU kernel time | not measured | not measured | not measured | not measured |

Shared donor assigned wall: 35.6246 hours; count once, not three times. The native campaign total is the shared pool plus all three continuation phases. Do not rank training efficiency against released-policy inference cost. Assigned wall includes overhead; it is not utilization-weighted kernel time. Partial SONIC observations are in sonic_cost.json, never extrapolated to 72 rows.

## All 12 Tasks: Full Horizon and Prefix

Each cell: full-motion completion; world mean / mean time-p95; local mean / mean time-p95 (cm). Tail cells show world / local last-20% means. Prefix tables repeat full-motion completion counts; they do not assert prefix completion as a replacement outcome.

### Nominal: Full Horizon

| Task | BeyondMimic | Static Search | MimicX | Released SONIC |
| --- | --- | --- | --- | --- |
| Turn and Jog | 6/6; W 8.4944 / 13.8025; L 3.1221 / 4.5969 | 6/6; W 10.2331 / 18.4792; L 3.1959 / 4.7719 | 6/6; W 9.3839 / 14.4007; L 3.0716 / 4.6356 | 3/3; W 12.6865 / 18.0719; L 2.2311 / 2.8944 |
| Side Step | 6/6; W 7.1142 / 8.3684; L 2.3058 / 4.2440 | 6/6; W 5.2596 / 7.5260; L 2.1195 / 3.9214 | 6/6; W 7.1142 / 8.3684; L 2.3058 / 4.2440 | 3/3; W 18.7966 / 39.1012; L 2.2346 / 3.2763 |
| Forward Jog | 6/6; W 4.9401 / 7.0609; L 1.7916 / 2.4546 | 6/6; W 4.6017 / 6.5913; L 1.9617 / 2.8291 | 6/6; W 5.4151 / 7.5201; L 1.7948 / 2.5762 | 3/3; W 4.8689 / 7.4603; L 2.4206 / 3.3497 |
| Rubbing Eyes | 6/6; W 1.6376 / 2.1868; L 1.3916 / 2.1002 | 6/6; W 1.6375 / 2.1867; L 1.3915 / 2.1002 | 6/6; W 1.6330 / 2.1866; L 1.3912 / 2.0950 | 3/3; W 2.7710 / 3.7557; L 2.1201 / 3.2448 |
| Praying/Salute | 6/6; W 3.4595 / 5.9034; L 2.5630 / 3.6112 | 6/6; W 3.2742 / 5.1387; L 2.6929 / 3.8344 | 6/6; W 3.3364 / 5.7055; L 2.6083 / 3.6280 | 3/3; W 7.6284 / 11.0462; L 3.3610 / 5.0773 |
| Heel-and-Toe Dance | 6/6; W 7.3059 / 12.5042; L 3.1729 / 4.5859 | 6/6; W 7.9073 / 16.7060; L 3.3987 / 4.8152 | 6/6; W 6.4493 / 10.9934; L 3.2269 / 4.3672 | 3/3; W 22.1936 / 38.9448; L 3.6176 / 5.0395 |
| Crouch Transition | 6/6; W 2.6653 / 3.6214; L 3.3086 / 3.7736 | 6/6; W 2.1078 / 3.0354; L 2.6846 / 3.3071 | 6/6; W 2.0551 / 2.9025; L 2.7408 / 3.4667 | 3/3; W 10.3987 / 12.9444; L 6.2596 / 6.8601 |
| Pointing Down | 6/6; W 2.9611 / 4.9420; L 1.2996 / 1.7541 | 6/6; W 3.0531 / 5.4660; L 1.3359 / 1.7428 | 6/6; W 3.0712 / 5.4239; L 1.3381 / 1.7357 | 3/3; W 3.1368 / 4.3978; L 1.8489 / 2.3813 |
| Dodge Backward | 6/6; W 5.9301 / 9.3140; L 2.3062 / 4.2446 | 6/6; W 7.6567 / 11.5441; L 2.5044 / 4.4686 | 6/6; W 5.7779 / 9.0926; L 2.3484 / 4.3042 | 3/3; W 25.5502 / 36.3314; L 2.7422 / 4.5906 |
| Injured-Torso Walk | 6/6; W 2.8219 / 4.3929; L 1.5668 / 2.1924 | 6/6; W 2.7140 / 4.1876; L 1.6205 / 2.2262 | 6/6; W 2.6992 / 4.0772; L 1.5310 / 2.1525 | 3/3; W 6.9633 / 12.0531; L 3.0234 / 3.7378 |
| Broad Jump | 0/6; W undefined / undefined; L undefined / undefined | 6/6; W 8.0376 / 21.7844; L 4.6328 / 8.6171 | 6/6; W 5.5600 / 11.1309; L 4.7990 / 8.7218 | 0/3; W undefined / undefined; L undefined / undefined |
| Get-up | 0/6; W undefined / undefined; L undefined / undefined | 0/6; W undefined / undefined; L undefined / undefined | 0/6; W undefined / undefined; L undefined / undefined | 0/3; W undefined / undefined; L undefined / undefined |

### Nominal: Common Prefix Only

| Task | BeyondMimic | Static Search | MimicX | Released SONIC |
| --- | --- | --- | --- | --- |
| Turn and Jog | 6/6; W 5.0684 / 9.3573; L 3.6984 / 4.9851 | 6/6; W 5.7191 / 12.5669; L 4.0750 / 5.3755 | 6/6; W 5.2777 / 9.3490; L 3.8779 / 5.4622 | 3/3; W 9.7787 / 17.8449; L 2.1868 / 2.7910 |
| Side Step | 6/6; W 6.3047 / 7.0607; L 2.8029 / 5.3021 | 6/6; W 4.4593 / 5.6843; L 2.3890 / 4.5305 | 6/6; W 6.3047 / 7.0607; L 2.8029 / 5.3021 | 3/3; W 6.6963 / 8.3576; L 1.9387 / 2.8300 |
| Forward Jog | 6/6; W 3.7487 / 5.3927; L 1.9071 / 2.4674 | 6/6; W 3.3851 / 4.8801; L 2.1270 / 2.9108 | 6/6; W 4.2094 / 5.9518; L 1.8360 / 2.5905 | 3/3; W 4.3055 / 7.3439; L 2.4548 / 3.3656 |
| Rubbing Eyes | 6/6; W 1.4301 / 2.1348; L 1.3844 / 2.0917 | 6/6; W 1.4301 / 2.1348; L 1.3844 / 2.0917 | 6/6; W 1.4303 / 2.1348; L 1.3846 / 2.0846 | 3/3; W 3.0271 / 3.9250; L 2.3454 / 3.5678 |
| Praying/Salute | 6/6; W 4.4185 / 6.1641; L 2.9771 / 3.7664 | 6/6; W 3.5742 / 5.0781; L 3.0130 / 3.7803 | 6/6; W 3.7284 / 5.1282; L 3.0912 / 3.7775 | 3/3; W 5.8542 / 8.5267; L 4.4699 / 8.0783 |
| Heel-and-Toe Dance | 6/6; W 3.9133 / 5.7894; L 2.3862 / 2.8953 | 6/6; W 3.1512 / 4.2399; L 2.5563 / 3.1114 | 6/6; W 2.8907 / 4.0641; L 3.0561 / 3.4641 | 3/3; W 3.1764 / 3.9889; L 2.8778 / 3.4684 |
| Crouch Transition | 6/6; W 2.6882 / 3.3990; L 3.2466 / 3.7646 | 6/6; W 1.7782 / 2.2292; L 2.5447 / 2.9402 | 6/6; W 1.7508 / 2.3663; L 2.5873 / 3.0935 | 3/3; W 9.9550 / 13.5192; L 5.7569 / 6.8591 |
| Pointing Down | 6/6; W 2.4189 / 3.6827; L 0.9283 / 1.0884 | 6/6; W 2.4392 / 3.6699; L 0.9196 / 1.0967 | 6/6; W 2.4392 / 3.6699; L 0.9196 / 1.0967 | 3/3; W 1.7961 / 3.7560; L 1.6439 / 3.0330 |
| Dodge Backward | 6/6; W 3.4297 / 5.3496; L 2.8152 / 3.7371 | 6/6; W 3.7959 / 5.7216; L 2.9517 / 4.1194 | 6/6; W 3.0771 / 5.2335; L 2.7045 / 3.5503 | 3/3; W 5.5393 / 8.8695; L 3.7727 / 4.5699 |
| Injured-Torso Walk | 6/6; W 2.3164 / 3.7993; L 1.5504 / 2.0745 | 6/6; W 1.6089 / 2.1561; L 1.5734 / 2.1321 | 6/6; W 1.9543 / 2.7784; L 1.5209 / 2.0915 | 3/3; W 3.9792 / 6.2135; L 3.6162 / 5.1838 |
| Broad Jump | 0/6; W undefined / undefined; L undefined / undefined | 6/6; W 2.7165 / 5.1679; L 2.4773 / 3.3334 | 6/6; W 3.7025 / 7.1683; L 2.7422 / 3.8827 | 0/3; W 5.6988 / 13.6726; L 5.5928 / 11.9594 |
| Get-up | 0/6; W 5.3126 / 7.2810; L 3.8183 / 4.7187 | 0/6; W 7.5738 / 10.4056; L 4.5581 / 5.6034 | 0/6; W 7.5738 / 10.4056; L 4.5581 / 5.6034 | 0/3; W 3.2558 / 6.7849; L 1.9509 / 5.6249 |

### Nominal: Last 20% Tail

| Task | BeyondMimic | Static Search | MimicX | Released SONIC |
| --- | --- | --- | --- | --- |
| Turn and Jog | 11.0001 / 2.2845 | 12.4506 / 2.5290 | 12.7566 / 2.3793 | 11.2546 / 1.9434 |
| Side Step | 7.2465 / 1.6864 | 5.3202 / 1.9594 | 7.2465 / 1.6864 | 39.0149 / 1.8977 |
| Forward Jog | 5.7070 / 1.5161 | 5.7625 / 1.6168 | 6.3168 / 1.5602 | 6.3959 / 2.6451 |
| Rubbing Eyes | 1.5756 / 1.6858 | 1.5754 / 1.6853 | 1.5704 / 1.6843 | 1.7816 / 1.8217 |
| Praying/Salute | 2.3949 / 1.6441 | 2.9193 / 1.8190 | 2.3495 / 1.8573 | 6.2285 / 1.0845 |
| Heel-and-Toe Dance | 5.5368 / 2.4185 | 6.0744 / 2.6740 | 5.7515 / 2.8786 | 32.0666 / 3.0713 |
| Crouch Transition | 1.9658 / 3.0908 | 2.5179 / 2.3002 | 2.5947 / 2.6873 | 11.6724 / 6.7090 |
| Pointing Down | 2.4055 / 1.3753 | 2.3265 / 1.3976 | 2.3976 / 1.3995 | 3.1102 / 1.6980 |
| Dodge Backward | 6.3141 / 1.5844 | 8.3211 / 1.9107 | 5.9383 / 1.7672 | 36.1025 / 1.6549 |
| Injured-Torso Walk | 2.4468 / 1.5603 | 2.6450 / 1.6972 | 2.5505 / 1.4885 | 11.1748 / 2.9982 |
| Broad Jump | undefined (censored) | 6.0076 / 5.4098 | 5.0896 / 5.1355 | undefined (censored) |
| Get-up | undefined (censored) | undefined (censored) | undefined (censored) | undefined (censored) |

### Perturbed: Full Horizon

| Task | BeyondMimic | Static Search | MimicX | Released SONIC |
| --- | --- | --- | --- | --- |
| Turn and Jog | 6/6; W 4.8854 / 9.1451; L 3.2011 / 4.5840 | 6/6; W 5.8216 / 12.1146; L 3.2503 / 4.7903 | 6/6; W 5.2603 / 9.3130; L 3.1547 / 4.6445 | 3/3; W 12.5320 / 17.9873; L 2.2274 / 2.8852 |
| Side Step | 6/6; W 7.1833 / 8.6521; L 2.4560 / 4.2513 | 6/6; W 5.2127 / 7.0348; L 2.1780 / 4.0341 | 6/6; W 7.1833 / 8.6521; L 2.4560 / 4.2513 | 3/3; W 19.1256 / 40.3753; L 2.2403 / 3.2933 |
| Forward Jog | 6/6; W 5.0888 / 7.2331; L 1.7118 / 2.3647 | 6/6; W 5.7794 / 8.7455; L 1.8560 / 2.6146 | 6/6; W 4.7263 / 6.6463; L 1.7380 / 2.4802 | 3/3; W 4.8407 / 7.2590; L 2.4191 / 3.3267 |
| Rubbing Eyes | 6/6; W 1.4255 / 1.8250; L 1.4313 / 2.1208 | 6/6; W 1.4256 / 1.8250; L 1.4313 / 2.1208 | 6/6; W 1.4246 / 1.8250; L 1.4310 / 2.1211 | 3/3; W 2.8483 / 4.2255; L 2.1887 / 3.6222 |
| Praying/Salute | 6/6; W 3.5613 / 5.6509; L 2.5434 / 3.5448 | 6/6; W 3.3125 / 4.9627; L 2.7061 / 3.8551 | 6/6; W 3.3571 / 5.3921; L 2.6065 / 3.6217 | 3/3; W 6.7308 / 9.9446; L 3.3274 / 5.1342 |
| Heel-and-Toe Dance | 6/6; W 6.5727 / 9.8620; L 3.1617 / 4.5271 | 6/6; W 7.6567 / 13.2136; L 3.3624 / 4.7600 | 6/6; W 5.5529 / 8.8455; L 3.2526 / 4.3382 | 3/3; W 18.4112 / 38.6652; L 3.6263 / 5.0977 |
| Crouch Transition | 6/6; W 2.9147 / 3.8294; L 3.4232 / 3.9066 | 6/6; W 2.1416 / 2.9628; L 2.8006 / 3.4035 | 6/6; W 2.0812 / 2.9721; L 2.8874 / 3.6608 | 3/3; W 11.0910 / 13.6958; L 6.2637 / 6.9250 |
| Pointing Down | 6/6; W 3.0542 / 5.2442; L 1.3773 / 1.7620 | 6/6; W 3.1452 / 5.9058; L 1.4017 / 1.7639 | 6/6; W 3.1392 / 5.9048; L 1.4006 / 1.7622 | 3/3; W 3.3101 / 4.7278; L 1.8616 / 2.3770 |
| Dodge Backward | 6/6; W 5.1739 / 9.0094; L 2.3535 / 4.3151 | 6/6; W 6.6941 / 10.9323; L 2.5529 / 4.5536 | 6/6; W 4.8990 / 8.8055; L 2.3748 / 4.2488 | 3/3; W 25.0434 / 35.6025; L 2.7411 / 4.5885 |
| Injured-Torso Walk | 6/6; W 3.0158 / 4.6264; L 1.4975 / 2.1421 | 6/6; W 2.9646 / 4.4925; L 1.5556 / 2.1654 | 6/6; W 2.7977 / 4.2731; L 1.4679 / 2.1278 | 3/3; W 7.6618 / 13.5025; L 3.0258 / 3.7437 |
| Broad Jump | 0/6; W undefined / undefined; L undefined / undefined | 6/6; W 7.9953 / 19.3282; L 4.5962 / 8.4331 | 6/6; W 5.3923 / 10.2338; L 4.8566 / 8.5412 | 0/3; W undefined / undefined; L undefined / undefined |
| Get-up | 0/6; W undefined / undefined; L undefined / undefined | 0/6; W undefined / undefined; L undefined / undefined | 0/6; W undefined / undefined; L undefined / undefined | 0/3; W undefined / undefined; L undefined / undefined |

### Perturbed: Common Prefix Only

| Task | BeyondMimic | Static Search | MimicX | Released SONIC |
| --- | --- | --- | --- | --- |
| Turn and Jog | 6/6; W 4.8366 / 7.4373; L 3.7558 / 5.2982 | 6/6; W 4.9722 / 8.9049; L 4.0578 / 5.6741 | 6/6; W 5.0132 / 7.9763; L 3.9142 / 5.7537 | 3/3; W 9.7558 / 17.7537; L 2.1778 / 2.7562 |
| Side Step | 6/6; W 6.2348 / 6.9980; L 2.9712 / 5.3728 | 6/6; W 4.4056 / 5.6629; L 2.4636 / 4.5782 | 6/6; W 6.2348 / 6.9980; L 2.9712 / 5.3728 | 3/3; W 6.6019 / 8.3017; L 1.9218 / 2.8152 |
| Forward Jog | 6/6; W 3.4751 / 4.8552; L 1.7993 / 2.2952 | 6/6; W 3.4361 / 5.0263; L 2.0269 / 2.6869 | 6/6; W 3.7048 / 5.3481; L 1.7574 / 2.4217 | 3/3; W 4.2756 / 7.2722; L 2.4374 / 3.3167 |
| Rubbing Eyes | 6/6; W 1.3373 / 2.0192; L 1.4078 / 2.0657 | 6/6; W 1.3373 / 2.0192; L 1.4078 / 2.0657 | 6/6; W 1.3373 / 2.0192; L 1.4078 / 2.0660 | 3/3; W 3.2667 / 4.5852; L 2.5432 / 4.2225 |
| Praying/Salute | 6/6; W 4.2734 / 5.9360; L 2.9867 / 3.8054 | 6/6; W 3.3492 / 4.7400; L 2.9734 / 3.7020 | 6/6; W 3.5448 / 4.9174; L 3.0792 / 3.7575 | 3/3; W 5.7980 / 7.8839; L 4.2396 / 7.1844 |
| Heel-and-Toe Dance | 6/6; W 4.0484 / 6.2943; L 2.4345 / 2.8747 | 6/6; W 3.6136 / 5.8523; L 2.5689 / 3.1434 | 6/6; W 3.0505 / 4.4563; L 3.1646 / 3.4633 | 3/3; W 3.1570 / 4.5787; L 2.7209 / 3.3998 |
| Crouch Transition | 6/6; W 2.8803 / 3.6891; L 3.3092 / 3.8109 | 6/6; W 1.7992 / 2.2575; L 2.6124 / 2.9619 | 6/6; W 1.8114 / 2.4070; L 2.6682 / 3.1332 | 3/3; W 10.2526 / 13.8128; L 5.7719 / 6.9405 |
| Pointing Down | 6/6; W 2.1509 / 3.1816; L 1.0173 / 1.1973 | 6/6; W 2.1738 / 3.1788; L 0.9725 / 1.1583 | 6/6; W 2.1738 / 3.1788; L 0.9725 / 1.1583 | 3/3; W 1.8514 / 3.7512; L 1.6584 / 3.0565 |
| Dodge Backward | 6/6; W 3.3415 / 5.4962; L 2.7650 / 3.8032 | 6/6; W 3.7124 / 5.9057; L 2.9951 / 4.2361 | 6/6; W 3.1580 / 5.5499; L 2.6985 / 3.5460 | 3/3; W 5.9276 / 9.7925; L 3.7872 / 4.5778 |
| Injured-Torso Walk | 6/6; W 2.3019 / 3.7804; L 1.5026 / 1.9958 | 6/6; W 1.6787 / 2.3133; L 1.5322 / 2.0481 | 6/6; W 1.9312 / 2.8223; L 1.5400 / 2.1817 | 3/3; W 3.9612 / 6.2387; L 3.6211 / 5.2229 |
| Broad Jump | 0/6; W undefined / undefined; L undefined / undefined | 6/6; W 2.8187 / 5.4605; L 2.5773 / 3.3858 | 6/6; W 3.8048 / 7.3923; L 2.8061 / 3.9018 | 0/3; W 5.7354 / 13.4915; L 5.5224 / 11.9300 |
| Get-up | 0/6; W 5.2844 / 7.5524; L 3.9717 / 4.8220 | 0/6; W 7.4645 / 10.6721; L 4.5846 / 5.6102 | 0/6; W 7.4645 / 10.6721; L 4.5846 / 5.6102 | 0/3; W 3.2325 / 6.7343; L 1.9498 / 5.6287 |

### Perturbed: Last 20% Tail

| Task | BeyondMimic | Static Search | MimicX | Released SONIC |
| --- | --- | --- | --- | --- |
| Turn and Jog | 4.3663 / 2.4078 | 5.1236 / 2.6332 | 4.7611 / 2.4848 | 11.2154 / 1.9436 |
| Side Step | 7.5080 / 1.9331 | 5.1576 / 1.9619 | 7.5080 / 1.9331 | 40.2946 / 1.9344 |
| Forward Jog | 6.3653 / 1.4647 | 7.5422 / 1.5617 | 6.0858 / 1.6058 | 6.2718 / 2.6434 |
| Rubbing Eyes | 1.4375 / 1.7404 | 1.4376 / 1.7404 | 1.4334 / 1.7399 | 1.8076 / 1.8328 |
| Praying/Salute | 2.6791 / 1.6659 | 3.3011 / 1.8192 | 2.6433 / 1.8919 | 3.7803 / 1.1077 |
| Heel-and-Toe Dance | 6.3125 / 2.4687 | 7.6782 / 2.7052 | 6.0015 / 2.9661 | 24.7883 / 3.0720 |
| Crouch Transition | 2.3492 / 3.2737 | 2.5385 / 2.4711 | 2.4889 / 2.8624 | 12.1521 / 6.6966 |
| Pointing Down | 2.5991 / 1.5007 | 2.4346 / 1.4634 | 2.4166 / 1.4594 | 3.3908 / 1.7314 |
| Dodge Backward | 4.9838 / 1.7994 | 6.5792 / 2.0342 | 4.5872 / 1.9193 | 35.3606 / 1.6558 |
| Injured-Torso Walk | 2.9909 / 1.4357 | 3.2483 / 1.6030 | 3.1295 / 1.3855 | 12.4855 / 3.0215 |
| Broad Jump | undefined (censored) | 6.5218 / 5.2570 | 4.8751 / 5.2217 | undefined (censored) |
| Get-up | undefined (censored) | undefined (censored) | undefined (censored) | undefined (censored) |
