# RigExpert

RigExpert is a tool designed to analyze and visualize antenna performance. It takes measurement data from devices like the RigExpert antenna analyzer or NanoVNA and generates detailed charts that help users understand their antenna's behavior.

These charts include impedance plots, VSWR (Voltage Standing Wave Ratio) graphs, and Smith charts, which are essential for fine-tuning and optimizing antenna systems.


### Example:

```
% RigExpert -t "Mike's Delta loop 20" -f ~/Downloads/M2.s1p --range 12:33 -A -D
08/30/24 08:13:40 - 242 INFO - /tmp/mike-s-delta-loop-20-all.png
08/30/24 08:13:40 - 225 INFO - /tmp/mike-s-delta-loop-20-dual.png
```
<img src="examples/mike-s-delta-loop-20-all.png" width="940">
<img src="examples/mike-s-delta-loop-20-dual.png" width="940">
