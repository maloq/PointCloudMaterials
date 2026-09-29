# Failed coordinate-analysis attempt

Job 1013457 stopped at its repeatability check before exporting valid metrics.
It required bitwise equality between a 256-example batch and a 64-example batch.
The successful revision verified exact matched-batch equality and separately
measured small batch-shape floating-point differences.

[Completed revision 2](../coordinate-profiles-v2/README.md). The original code,
configuration and log are preserved under technical/.
